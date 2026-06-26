"""FastAPI app factory for the worker dashboard.

Five HTML pages (Overview, Activity, Models, Receipts, Config) plus a small JSON
`/api/stats` endpoint that drives the live poll. Mostly read-only views, with a
few localhost-only controls — pause/resume, account login/logout, and the
execution-policy setter; the daemon's existing threads remain the writers for
everything else.

The app captures the worker's local SQLite state DB + the WorkerConfig
at construction time and reads from them on each request.
"""

from __future__ import annotations

import html
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from auspexai_worker import __version__
from auspexai_worker.accelerator import detect_accelerator
from auspexai_worker.config import WorkerConfig
from auspexai_worker.health import ThermalMonitor, ThermalState
from auspexai_worker.models import ModelStore
from auspexai_worker.state import (
    AssignmentAuditRepository,
    Database,
    PendingSubmissionRepository,
    SubmittedResultRepository,
    TenantListRepository,
    WorkerSelfRepository,
)
from auspexai_worker.worker_state import derive_self_state
from auspexai_worker.workspace import workspace_runs_dir

from .templates import render_cards, render_kv, render_page, render_table

_LOG = logging.getLogger(__name__)


class _LoginSnapshot(NamedTuple):
    status: str
    user_code: str | None
    verification_uri: str | None
    error: str | None


class _LoginSession:
    """Thread-safe state for an in-progress dashboard login (GitHub Device Flow).

    The Device Flow is multi-step and BLOCKS (it polls GitHub until the volunteer
    authorizes), so it can't be a single POST like logout. The POST /login route
    kicks off a background thread that drives the flow; this object is the bridge
    the thread writes to and the GET /login page reads from. One instance per app.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = "idle"  # 'idle' | 'pending' | 'authorized' | 'failed'
        self._user_code: str | None = None
        self._verification_uri: str | None = None
        self._error: str | None = None

    def start(self) -> None:
        with self._lock:
            self._status = "pending"
            self._user_code = None
            self._verification_uri = None
            self._error = None

    def set_code(self, user_code: str, verification_uri: str) -> None:
        with self._lock:
            self._user_code = user_code
            self._verification_uri = verification_uri

    def set_authorized(self) -> None:
        with self._lock:
            self._status = "authorized"

    def set_failed(self, error: str) -> None:
        with self._lock:
            self._status = "failed"
            self._error = error

    def reset(self) -> None:
        """Return to idle — consume a completed flow so the one-time citation prompt
        doesn't re-show once the volunteer has made (or skipped) the choice."""
        with self._lock:
            self._status = "idle"
            self._user_code = None
            self._verification_uri = None
            self._error = None

    def snapshot(self) -> _LoginSnapshot:
        with self._lock:
            return _LoginSnapshot(
                status=self._status,
                user_code=self._user_code,
                verification_uri=self._verification_uri,
                error=self._error,
            )


def _fmt_relative(when: datetime | None) -> str:
    """Render a datetime as 'Nm ago' or 'never'."""
    if when is None:
        return "never"
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    delta = now - when
    secs = int(delta.total_seconds())
    if secs < 0:
        return when.isoformat()
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _tier_badge(tier: int) -> str:
    names = {
        0: "T0 anonymous",
        1: "T1 authenticated",
        2: "T2 trusted",
        3: "T3 vetted",
    }
    label = html.escape(names.get(tier, f"T{tier}"))
    return f'<span class="badge tier-{tier}">{label}</span>'


def _executor_indicator(policy: str) -> str:
    """The code-execution mode as a calm dot + word (+ tooltip) — shared by the
    overview Details and the Config "current" line.

    The dot is NEUTRAL: executor mode is a configuration, not a health status, so
    it carries no good/warn/bad color. In particular provisioned is NOT amber —
    running provisioned tenant code is a deliberate, consented operating mode, not
    a warning. The explicit word + the tooltip carry the §5.14 consent awareness;
    the loud, deliberate-act friction lives on the /executor/confirm enable step."""
    tip = {
        "synthetic": "Synthetic only — no third-party code runs.",
        "provisioned": "Runs provisioned tenant code — you consented to run third-party code here.",
        "off": "Off — this worker refuses all work.",
    }.get(policy, "")
    return f'<span class="dot"></span><span title="{html.escape(tip)}">{html.escape(policy)}</span>'


def _current_executor_policy(config: WorkerConfig, config_path: Path | None) -> str:
    """The live executor policy the daemon enforces — read fresh from worker.toml
    (the source of truth) so the dashboard reflects a change immediately. The daemon
    hot-reloads the same value (per dispatch + per heartbeat), so there's no
    stale-snapshot / pending-restart split. Falls back to the daemon-start snapshot
    if the file can't be read."""
    try:
        return WorkerConfig.load(config_path=config_path).execute_tenant_code
    except Exception:
        return config.execute_tenant_code


def _thermal_monitor(config: WorkerConfig) -> ThermalMonitor:
    return ThermalMonitor(
        warn_c=config.thermal_warn_c,
        crit_c=config.thermal_crit_c,
        resume_c=config.thermal_resume_c,
    )


def _thermal_html(config: WorkerConfig) -> str:
    mon = _thermal_monitor(config)
    if not mon.enabled:
        return '<span class="muted">no thermal sensor — governor inactive on this host</span>'
    snap = mon.snapshot()
    cls = {ThermalState.OK: "ok", ThermalState.WARM: "warn", ThermalState.CRITICAL: "error"}[
        snap.state
    ]
    temp = f"{snap.current_temp_c}°C" if snap.current_temp_c is not None else "—"
    return f'{html.escape(temp)} <span class="badge {cls}">{snap.state.value}</span>'


def _thermal_critical(config: WorkerConfig) -> bool:
    """True when the host is thermal-CRITICAL (auto-refusing work). WARM is
    advisory and does not count. Off where there's no sensor."""
    mon = _thermal_monitor(config)
    return mon.enabled and mon.snapshot().state is ThermalState.CRITICAL


def _inflight_unit(runs_dir: Path) -> str | None:
    """The unit currently executing (a workspace with a LIVE runner pid), or
    None. The dispatcher writes runner.pid per workspace and cleans the
    workspace on completion, so a live pid == a unit running right now."""
    import os

    if not runs_dir.is_dir():
        return None
    for ws in sorted(runs_dir.iterdir()):
        pid_file = ws / "runner.pid"
        if not pid_file.is_file():
            continue
        try:
            pid = int(pid_file.read_text(encoding="ascii").strip())
            os.kill(pid, 0)  # liveness probe; raises if no such process
        except (ValueError, OSError):
            continue
        return ws.name
    return None


def _work_activity(
    config: WorkerConfig,
    *,
    runs_dir: Path,
    last_submitted_at: datetime | None,
    now: datetime,
) -> tuple[str, str]:
    """The accurate, dynamic (headline, detail) for an ACTIVE worker — only
    claims 'Receiving work' when work is genuinely flowing. The headline
    replaces the generic 'active' label so the banner reads as the actual
    activity. Signals (most→least specific): a live runner = a unit running
    now; a recently-submitted unit = receiving work; otherwise idle/available
    (no overclaim)."""
    if _inflight_unit(runs_dir) is not None:
        return "Running a work unit", "executing now"
    # "recently" tracks the assignment cadence — a few poll intervals, floored
    # so a slow poll config doesn't make a busy worker read as idle.
    window = max(3 * config.assignment_poll_interval_seconds, 180)
    if last_submitted_at is not None:
        ts = last_submitted_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if (now - ts).total_seconds() <= window:
            return "Receiving work", f"last unit completed {_fmt_relative(ts)}"
        return "Idle", f"available; no work assigned recently (last unit {_fmt_relative(ts)})"
    return (
        "Idle",
        "available; no work has been assigned yet — the network has no matching work for it right now",
    )


def _update_notice(worker, config: WorkerConfig) -> tuple[str, str]:
    """§9 #46: (css_class, inner_html) for the update-available notice, or
    ('', '') when the worker is current / nothing announced. Quiet and
    persistent (no dismissal state to manage) — display-time version
    comparison hides it the moment the worker upgrades. The headline is
    coordinator-supplied text: treat as UNTRUSTED display input — escaped
    here, truncated at the wire parse.

    The command is PRINTED + one-click COPIED, never run: the dashboard's
    mutating controls only write config values; executing a network-fetched
    installer (which needs sudo) from a localhost HTTP endpoint would be a
    different class of surface entirely (§9 #41). The copy button keeps the
    volunteer's terminal as the executor — the click is the election, the
    paste is the consent."""
    from auspexai_worker import __version__
    from auspexai_worker.updates import is_newer_version, upgrade_command

    latest = getattr(worker, "latest_release_version", None) if worker else None
    if not latest or not is_newer_version(latest, __version__):
        return "", ""
    notes = getattr(worker, "latest_release_notes", None)
    url = getattr(worker, "latest_release_url", None)
    parts = [f"<strong>Update available: v{html.escape(latest)}</strong>"]
    if notes:
        parts.append(f"— {html.escape(notes)}")
    link = ""
    if url and url.startswith("https://"):
        link = f' (<a href="{html.escape(url)}" target="_blank" rel="noreferrer">release notes</a>)'
    cmd = html.escape(upgrade_command(config.flavor))
    copy_btn = (
        f'<button type="button" class="copy-cmd" data-cmd="{cmd}" '
        'onclick="navigator.clipboard.writeText(this.dataset.cmd)'
        ".then(()=>{this.textContent='copied!';"
        "setTimeout(()=>{this.textContent='copy command';},2000);})\">"
        "copy command</button>"
    )
    inner = (
        " ".join(parts) + f"{link}<br>To upgrade, paste this in a terminal: "
        f"<code>{cmd}</code> {copy_btn}<br>"
        '<span class="muted">Updates are never automatic — upgrading is always your choice; '
        "your enrollment, keys, and models survive the upgrade.</span>"
    )
    return "notice", inner


def _inference_status(config: WorkerConfig) -> dict[str, Any] | None:
    """W-S: live reachability of the inference backend, or None when
    `[inference] backend = "none"` (not an inference host). The heart renders it
    as a vital (dot), like the coordinator-connection vital."""
    backend = getattr(config, "inference_backend", "none")
    if backend == "none":
        return None
    if backend == "ollama":
        version: str | None = None
        try:
            from auspexai_worker.inference import OllamaBackend

            be = OllamaBackend(config.inference_ollama_url)
            healthy = be.is_healthy()
            if healthy:
                version = be.version()
        except Exception:
            healthy = False
        return {
            "backend": "ollama",
            "reachable": bool(healthy),
            "version": version,  # §9 #46 determinism provenance
            "url": config.inference_ollama_url,
        }
    return {"backend": str(backend), "reachable": True, "version": None, "url": None}


def build_app(*, db: Database, config: WorkerConfig, config_path: Path | None = None) -> FastAPI:
    """Build the dashboard FastAPI app.

    Args:
        db: opened state DB. Reused across requests; SQLite WAL mode
            and the worker's re-entrant transaction lock make this
            safe alongside the daemon's writer threads.
        config: snapshot of the loaded WorkerConfig. Read-only for
            the dashboard's display; the M9 leg-4 executor setter is the
            one write path, and it writes the TOML at `config_path` (the
            change applies on the next daemon restart, like the CLI).
        config_path: the worker.toml the executor setter writes to. Defaults
            to the standard XDG path when not supplied.
    """
    from auspexai_worker.config import default_worker_toml_path, set_executor_policy

    toml_path = config_path or default_worker_toml_path()
    # The accelerator is static hardware — detect it ONCE at build (its probe
    # shells out to nvidia-smi / reads /dev), not on every /api/stats poll.
    accelerator_label = detect_accelerator().label
    app = FastAPI(
        title="AuspexAI Worker — local dashboard",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def _no_store(request: Request, call_next):
        # The page is server-rendered, so a daemon upgrade only takes effect on a
        # page reload; no-store guarantees that reload fetches the NEW HTML rather
        # than a browser-cached copy (the stale-layout gotcha after a roll).
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    self_repo = WorkerSelfRepository(db)
    audit_repo = AssignmentAuditRepository(db)
    results_repo = SubmittedResultRepository(db)
    pending_repo = PendingSubmissionRepository(db)
    tenant_lists = TenantListRepository(db)

    # One login session per app — the bridge between the background Device-Flow
    # thread (POST /login spawns it) and the GET /login page that polls for the
    # code + completion. Captured as a closure variable by the login routes.
    login_session = _LoginSession()

    @app.middleware("http")
    async def _citation_gate(request: Request, call_next):
        # One-time post-login citation gate: until a freshly-bound volunteer resolves
        # the /login/citation prompt, every dashboard GET bounces back to it — so the
        # choice can't be sidestepped by editing the URL to "/" (or any page). Exempt:
        # the login flow itself (/login*) and the stats poll. Only triggers in the
        # post-bind/pre-choice window (login_session pending/authorized); a long-bound
        # worker sits at idle and is unaffected.
        path = request.url.path
        if request.method == "GET" and not path.startswith("/login") and path != "/api/stats":
            w = self_repo.get()
            if (
                w is not None
                and w.account_binding_json is not None
                and login_session.snapshot().status in ("pending", "authorized")
            ):
                return RedirectResponse("/login/citation", status_code=303)
        return await call_next(request)

    def _gather_stats() -> dict[str, Any]:
        worker = self_repo.get()
        # Approximate counts via list-truncate-to-many. These are local-
        # only operations (SQLite query); acceptable for the dashboard
        # at Phase 2 closed-beta volume.
        receipts = results_repo.list_receipts(limit=10000)
        pending = pending_repo.list_all()
        audit = audit_repo.query(limit=10000)
        allow = tenant_lists.list_allow()
        deny = tenant_lists.list_deny()
        return {
            "worker": worker,
            "receipts_count": len(receipts),
            "pending_submissions": len(pending),
            "audit_count": len(audit),
            "tenant_allow_count": len(allow),
            "tenant_deny_count": len(deny),
        }

    # ---- routes ---------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def overview() -> str:
        stats = _gather_stats()
        worker = stats["worker"]
        if worker is None:
            body = (
                '    <p class="empty">This worker is not yet enrolled. '
                "Run <code>auspexai-worker bootstrap</code> from a terminal "
                "to enroll as T0 anonymous against the coordinator.</p>"
                f'    <p class="muted">Coordinator: '
                f"<code>{html.escape(config.coordinator_url)}</code></p>"
            )
            return render_page(title="Overview", body=body, active_nav="/")

        # ── Identity: who this worker IS — public key · enrolled · flavor. Its
        # OPERATIONAL state moved UP into the heart: how it computes (accelerator),
        # what it runs (execution), heat + inference are the heart's vitals; its
        # liveness is the heart's pulse; its metrics + tier ride the heart too.
        # Values share one type system: plain text, mono only for the pubkey.
        pubkey_short = f"{worker.pubkey_hex[:10]}…{worker.pubkey_hex[-8:]}"
        identity_rows: list[tuple[str, str, bool]] = [
            ("public key", f'<span title="{worker.pubkey_hex}">{pubkey_short}</span>', True),
            (
                "enrolled",
                f'<span title="{html.escape(worker.enrolled_at.isoformat())}">'
                f"{_fmt_relative(worker.enrolled_at)}</span>",
                False,
            ),
            *([("flavor", html.escape(config.flavor), False)] if config.flavor else []),
        ]
        identity_html = "    <h2>Identity</h2>\n" + render_cards(identity_rows)

        # Units completed + distinct experiments are the heart's headline metrics,
        # and receipts + pending are folded into the heart's metrics row too — so
        # there's no separate "Contribution ledger" section. (The old audit-row
        # count + tenant allow/deny cards were plumbing with no volunteer value;
        # they're off the overview — audit lives on Activity, allow/deny on Config.)
        progress = results_repo.progress_summary()

        upgrade_html = ""
        if (
            int(worker.trust_tier) == 0
            and config.upgrade_prompt_enabled
            and progress["completed_units"] >= config.upgrade_prompt_threshold
        ):
            upgrade_html = (
                '    <div class="notice">'
                "You've contributed enough to build a portable track record. "
                "Run <code>auspexai-worker login</code> to claim your contributions."
                "</div>\n"
            )

        # Worker state (active / idle / hold / fault) is the HEART's job now — the
        # redundant state banner was removed (one state surface). What remains is the
        # §9 #46 update-available notice; ALWAYS render the container (possibly empty)
        # so the live poll can flip it on without a page reload.
        notice_cls, notice_inner = _update_notice(worker, config)
        update_notice = (
            f'    <div class="{notice_cls}" data-live="update_notice">{notice_inner}</div>\n'
        )

        # The self-pause control is an ACTION (not status) — offered only where
        # the volunteer can actually act. An operator hold (pause OR quarantine)
        # is operator-controlled — don't dangle a pause/unpause that wouldn't
        # change the work state; if self-paused *under* an operator hold, say
        # resuming won't restore work until the hold lifts.
        # The pause/resume toggle is the work on/off switch — it lives in the
        # heart's lower-right, by the state it controls. Compact button; the longer
        # explanation rides the title tooltip.
        pause_control = ""
        if worker.self_paused:
            tip = "Resume receiving work."
            if worker.operator_hold_kind is not None:
                tip = (
                    "An operator hold is also active — resuming won't restore work "
                    "until the operator lifts it."
                )
            pause_control = (
                '<form method="post" action="/self-unpause" class="action">'
                f'<button type="submit" class="btn" title="{html.escape(tip)}">resume</button>'
                "</form>"
            )
        elif worker.operator_hold_kind is None:
            pause_control = (
                '<form method="post" action="/self-pause" class="action">'
                '<button type="submit" class="btn" '
                'title="Stop receiving work; keep your enrollment + tier.">pause</button>'
                "</form>"
            )

        # Log out / log in is an ACTION pair (mutually exclusive). Bound → offer
        # log out (drop the binding, revert to T0, keep the worker running — the
        # inverse of `login`). Anonymous → offer log in (bind a GitHub account to
        # build portable trust). Login is OFFERED whenever anonymous, independent
        # of the upgrade-prompt threshold — it parallels the CLI `login`.
        # Log out / log in is an ACCOUNT action (mutually exclusive) — it belongs
        # with Identity, not the activity heart. Styled button; the rationale rides
        # the title tooltip. Bound → offer log out (revert to T0, keep running);
        # anonymous → offer log in (bind a GitHub account for portable trust).
        account_control = ""
        if worker.account_binding_json is not None:
            account_control = (
                '<form method="post" action="/logout" class="action">'
                '<button type="submit" class="btn danger" '
                'title="Drop the GitHub-account binding and revert to T0; the worker '
                'keeps running. Re-bind any time with: auspexai-worker login.">'
                "log out</button></form>"
            )
        else:
            account_control = (
                '<form method="post" action="/login" class="action">'
                '<button type="submit" class="btn" '
                'title="Bind this worker to a GitHub account to build portable trust. '
                'Stays anonymous in citations unless you opt in separately.">'
                "log in</button></form>"
            )

        # The volunteer's heart monitor — "is my machine helping?" at a glance
        # (surface_liveness_and_activity_view_design.md). Skeleton rendered here;
        # the live poll fills the pulse + dot + narration on its immediate first
        # tick (no blank flash), then keeps it beating.
        tier_chip = _tier_badge(int(worker.trust_tier))
        heart_html = f"""    <section class="heart" id="wkr-heart">
      <header>
        <span class="pulse-dot" id="heart-dot"></span>
        <h2 class="heart-h">Activity</h2>
        {tier_chip}
      </header>
      <div class="heart-id">{html.escape(worker.worker_id)} · v{html.escape(__version__)}</div>
      <div class="strip" id="heart-strip"><span class="strip-empty">listening…</span></div>
      <p class="narration" id="heart-narration">—</p>
      <div class="heart-vitals" id="heart-vitals"></div>
      <div class="heart-foot">
        <div class="heart-metrics">
          <div class="hm"><span class="n" id="heart-units">{progress["completed_units"]}</span><span class="l">units contributed</span></div>
          <div class="hm"><span class="n" id="heart-exps">{progress["distinct_experiments"]}</span><span class="l">experiments</span></div>
          <div class="hm"><span class="n" data-live="receipts_count">{stats["receipts_count"]}</span><span class="l">receipts</span></div>
          <div class="hm"><span class="n" data-live="pending_submissions">{stats["pending_submissions"]}</span><span class="l">pending</span></div>
        </div>
        {pause_control}
      </div>
    </section>
"""

        body = (
            update_notice
            + heart_html
            + identity_html
            + "\n"
            + upgrade_html
            + f'\n    <div class="actions">{account_control}</div>\n'
        )
        return render_page(title="Overview", body=body, active_nav="/", live=True)

    @app.post("/self-pause")
    def self_pause() -> RedirectResponse:
        """§2.1 #11: the volunteer's own no-fault pause (low-risk; localhost-only).
        Takes effect within a heartbeat (the daemon declares self_paused + stops
        polling for work). No reason is collected — pausing your own box is the
        owner's prerogative, not something to justify."""
        if self_repo.get() is not None:
            self_repo.set_self_pause(True)
        return RedirectResponse("/", status_code=303)

    @app.post("/self-unpause")
    def self_unpause() -> RedirectResponse:
        if self_repo.get() is not None:
            self_repo.set_self_pause(False)
        return RedirectResponse("/", status_code=303)

    @app.post("/logout")
    def logout() -> RedirectResponse:
        """The inverse of `login` (mirrors the CLI `logout`): the coordinator reverts
        this worker to T0-anonymous and the local binding is cleared, but the worker
        stays enrolled and running. Localhost-only, like the other action routes.

        A coordinator-unreachable failure does NOT clear the local binding (the
        volunteer is still bound — try again); a worker the coordinator no longer
        knows is treated as already-unbound, so we clear locally anyway."""
        worker = self_repo.get()
        if worker is None or worker.account_binding_json is None:
            # Not enrolled, or already T0-anonymous: nothing to unbind.
            return RedirectResponse("/?logout=anon", status_code=303)

        # Lazy imports: the keystore/coordinator deps are only needed for this one
        # write path and shouldn't load on every dashboard render.
        from auspexai_worker.bootstrap import build_signer, open_keystore
        from auspexai_worker.coordinator.client import (
            CoordinatorClient,
            CoordinatorError,
            WorkerNotFoundError,
        )

        keystore = open_keystore(config)
        signer = build_signer(keystore)
        try:
            with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
                client.unbind_worker(worker_id=worker.worker_id)
        except WorkerNotFoundError:
            # Coordinator has no record of this worker; clear the local binding anyway.
            pass
        except CoordinatorError as exc:
            _LOG.warning("dashboard logout: coordinator unbind failed: %s", exc)
            return RedirectResponse("/?logout=failed", status_code=303)

        self_repo.update_after_unbind()
        return RedirectResponse("/?logout=ok", status_code=303)

    def _run_login() -> None:
        """Background-thread body for the dashboard login (GitHub Device Flow).

        Drives the full flow: blocks polling GitHub via `run_device_flow` (which
        calls `on_code` early so the /login page can show the code), then runs the
        same coordinator exchange + upgrade the CLI `login` does, and persists the
        binding. Binds ANONYMOUS — the public-attribution opt-in is a separate,
        deliberate choice (made later via the CLI / account controls), NOT here.

        Catches broadly so any failure surfaces in the UI (status='failed') rather
        than dying silently in the daemon thread."""
        # Lazy imports: the keystore/coordinator/oauth deps are only needed for
        # this one write path and shouldn't load on every dashboard render.
        from auspexai_worker.bootstrap import build_signer, open_keystore
        from auspexai_worker.coordinator.client import CoordinatorClient
        from auspexai_worker.oauth import run_device_flow

        try:
            access_token = run_device_flow(
                on_code=lambda c: login_session.set_code(c.user_code, c.verification_uri)
            )
            worker = self_repo.get()
            if worker is None or worker.account_binding_json is not None:
                # Enrollment vanished, or a concurrent bind already landed.
                login_session.set_authorized()
                return
            keystore = open_keystore(config)
            signer = build_signer(keystore)
            with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
                exchange = client.oauth_exchange(idp="github", access_token=access_token)
                status_after = client.upgrade_worker(
                    worker_id=worker.worker_id,
                    binding_token=exchange.binding_token,
                )
            binding_payload = json.dumps(
                {
                    "idp": "github",
                    "account_id": exchange.account_id,
                    "bound_at": exchange.expires_at.isoformat(),
                    "is_new_account": exchange.is_new_account,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            self_repo.update_after_upgrade(
                new_tier=status_after.trust_tier,
                account_binding_json=binding_payload,
            )
            login_session.set_authorized()
        except Exception as exc:  # surface in the UI, don't die silently in-thread
            _LOG.warning("dashboard login: device flow / upgrade failed: %s", exc)
            login_session.set_failed(str(exc))

    @app.post("/login")
    def login() -> RedirectResponse:
        """Kick off a GitHub Device-Flow login (the inverse of `logout`). Unlike
        logout, this can't be a single POST — the Device Flow is multi-step and
        blocks polling GitHub — so we spawn a background thread and redirect to the
        /login page, which polls for the code + completion. Localhost-only."""
        worker = self_repo.get()
        if worker is None or worker.account_binding_json is not None:
            # Not enrolled, or already bound: nothing to do.
            return RedirectResponse("/", status_code=303)
        if login_session.snapshot().status == "pending":
            # A flow is already running — don't start a second one.
            return RedirectResponse("/login", status_code=303)
        login_session.start()
        threading.Thread(
            target=_run_login,
            name="auspexai-worker-dashboard-login",
            daemon=True,
        ).start()
        return RedirectResponse("/login", status_code=303)

    @app.get("/login", response_class=HTMLResponse, response_model=None)
    def login_page() -> HTMLResponse | RedirectResponse:
        """The login waiting room: shows the user_code + verification URI and polls
        (meta-refresh) until the worker is bound. A standalone page using the
        dashboard's base layout."""
        worker = self_repo.get()
        bound = worker is not None and worker.account_binding_json is not None
        snap = login_session.snapshot()
        if bound:
            # Just bound via THIS flow → the one-time citation prompt, then the
            # dashboard. (snap is pending/authorized only during the flow that's
            # landing the bind; a previously-bound worker sits at idle and goes
            # straight to "/".)
            if snap.status in ("pending", "authorized"):
                return RedirectResponse("/login/citation", status_code=303)
            return RedirectResponse("/", status_code=303)
        if snap.status == "idle":
            return RedirectResponse("/", status_code=303)

        meta_refresh = '    <meta http-equiv="refresh" content="3">\n'
        if snap.status == "pending":
            if snap.user_code:
                uri = html.escape(snap.verification_uri or "")
                code = html.escape(snap.user_code)
                body = (
                    meta_refresh + "    <h2>Finish linking your GitHub account</h2>\n"
                    f'    <p>Open <a href="{uri}" target="_blank" rel="noreferrer">'
                    f"<code>{uri}</code></a> in a browser and enter this code:</p>\n"
                    f'    <p style="font-size:1.6em;letter-spacing:0.12em">'
                    f"<strong><code>{code}</code></strong></p>\n"
                    '    <p class="muted">Waiting for authorization… this page updates '
                    "automatically. Once you're linked, you'll choose whether to be "
                    "publicly credited in research citations.</p>\n"
                )
            else:
                body = (
                    meta_refresh + "    <h2>Starting GitHub login…</h2>\n"
                    '    <p class="muted">Requesting a device code from GitHub…</p>\n'
                )
            return HTMLResponse(render_page(title="Log in", body=body, active_nav="/"))

        if snap.status == "failed":
            err = html.escape(snap.error or "unknown error")
            body = (
                f'    <div class="notice fault">Login failed: {err}</div>\n'
                '    <form method="post" action="/login" style="margin:0.75em 0">'
                '<button type="submit">try again</button></form>\n'
                '    <p><a href="/">back to overview</a></p>\n'
            )
            return HTMLResponse(render_page(title="Log in", body=body, active_nav="/"))

        # 'authorized' but not yet reflected on the worker row (or a no-op finish).
        return RedirectResponse("/", status_code=303)

    # M9 leg 4: the executor-policy setter — the owner's code-execution consent
    # (synthetic/provisioned/off). This was deferred from the dashboard, not
    # permanently withheld. Localhost-only ⇒ only the box owner reaches it, but
    # enabling `provisioned` (consenting to run third-party code) is gated behind
    # an explicit confirm step so it stays a *deliberate act*; downgrading toward
    # a safer mode is one click. The daemon HOT-RELOADS the change (no restart) —
    # effective within one heartbeat. Tier-agnostic by design — owner consent is a
    # different axis than the network's trust tier (see M9 leg-4 design).
    @app.get("/executor/confirm", response_class=HTMLResponse)
    def executor_confirm() -> str:
        body = (
            "    <h2>Enable provisioned execution?</h2>\n"
            '    <div class="notice">Switching to <strong>provisioned</strong> means '
            "this machine will run <strong>third-party tenant code</strong> — but only "
            "executors that were operator-staged locally and whose hash matches the "
            "coordinator's manifest (anything else is refused, never echoed). This is "
            "your consent, on your hardware. It takes effect within one heartbeat "
            "(no restart needed).</div>\n"
            '    <form method="post" action="/executor" style="margin:1em 0">\n'
            '      <input type="hidden" name="policy" value="provisioned">\n'
            '      <input type="hidden" name="confirm" value="yes">\n'
            '      <button type="submit">Yes, enable provisioned execution</button>\n'
            '      <a href="/config" style="margin-left:1em">Cancel</a>\n'
            "    </form>\n"
        )
        return render_page(title="Confirm", body=body, active_nav="/config")

    @app.post("/executor", response_model=None)
    async def set_executor(request: Request) -> RedirectResponse | HTMLResponse:
        raw = (await request.body()).decode("utf-8", "replace")
        fields = parse_qs(raw)
        policy = (fields.get("policy", [""])[0] or "").strip()
        confirm = (fields.get("confirm", [""])[0] or "").strip()
        if policy not in ("synthetic", "provisioned", "off"):
            return RedirectResponse("/config", status_code=303)
        # Enabling provisioned requires the explicit confirm step.
        if policy == "provisioned" and confirm != "yes":
            return RedirectResponse("/executor/confirm", status_code=303)
        try:
            set_executor_policy(toml_path, policy)
        except OSError as e:
            body = (
                f'    <div class="notice">Could not write {html.escape(str(toml_path))}: '
                f"{html.escape(str(e))}</div>\n"
                '    <p><a href="/config">back to config</a></p>'
            )
            return HTMLResponse(render_page(title="Config", body=body, active_nav="/config"))
        # The daemon hot-reloads execute_tenant_code (per dispatch + per heartbeat),
        # so the change is live without a restart — just redirect back to /config,
        # which now reflects the new policy immediately.
        return RedirectResponse("/config", status_code=303)

    # System B (D): the public-citation opt-in as a ONE-TIME step in the login flow.
    # Authentication binds ANONYMOUS (login never asks — see `_run_login`); this is
    # the deliberate, separate choice the login defers to, presented on a single
    # ephemeral page right after the bind lands — and re-asked at EVERY login, since
    # logging out reverts the worker to anonymous (new work stays anonymous until the
    # next login). NOT a standing dashboard control. Localhost-only ⇒ box owner only.
    def _bound_account_id() -> str | None:
        w = self_repo.get()
        if w is None or w.account_binding_json is None:
            return None
        try:
            return json.loads(w.account_binding_json).get("account_id")
        except (ValueError, TypeError):
            return None

    @app.get("/login/citation", response_class=HTMLResponse, response_model=None)
    def login_citation() -> HTMLResponse | RedirectResponse:
        """The one-time post-login citation choice. Reached only right after a fresh
        bind (the login flow is pending/authorized); a revisit / non-fresh state goes
        straight to the dashboard. Defaults to ANONYMOUS — opt-in is explicit, and the
        question is re-asked at every login."""
        account_id = _bound_account_id()
        if account_id is None or login_session.snapshot().status not in ("pending", "authorized"):
            return RedirectResponse("/", status_code=303)
        body = (
            "    <h2>You're signed in — one quick choice.</h2>\n"
            '    <p class="muted">When research your compute helped run gets published, you can be '
            "named in its contributor acknowledgment — or stay anonymous. Your call, separate from "
            "signing in, and we ask again each time you log in. (While logged out, new work stays "
            "anonymous.)</p>\n"
            '    <form method="post" action="/login/citation" class="choice-form">\n'
            '      <fieldset class="choices">\n'
            '        <label class="choice"><input type="radio" name="choice" value="anonymous" '
            "checked> <span>Stay <strong>anonymous</strong> — no public credit</span></label>\n"
            '        <label class="choice"><input type="radio" name="choice" value="cite"> '
            "<span><strong>Credit me</strong> publicly, under my verified GitHub identity</span>"
            "</label>\n"
            '        <p class="muted choice-note">Credit always uses the GitHub login you signed '
            "in with — there's no free-text name, so a citation is always a real, verifiable "
            "identity.</p>\n"
            "      </fieldset>\n"
            '      <button type="submit" class="btn primary">Continue to my dashboard</button>\n'
            "    </form>\n"
        )
        return HTMLResponse(render_page(title="Citation", body=body, active_nav="/"))

    @app.post("/login/citation", response_model=None)
    async def login_citation_set(request: Request) -> RedirectResponse:
        """Capture the one-time choice (best-effort PUT), consume the login flow, and
        send the volunteer to their dashboard. A coordinator hiccup leaves them on the
        anonymous default — re-choosable by logging out and back in, or via the CLI
        `account attribution`."""
        account_id = _bound_account_id()
        if account_id is not None:
            raw = (await request.body()).decode("utf-8", "replace")
            fields = parse_qs(raw)
            public = fields.get("choice", ["anonymous"])[0] == "cite"
            from auspexai_worker.bootstrap import build_signer, open_keystore
            from auspexai_worker.coordinator.client import CoordinatorClient

            try:
                signer = build_signer(open_keystore(config))
                with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
                    client.set_attribution(
                        account_id=account_id,
                        public_attribution=public,
                        # Credit always uses the account's verified GitHub login
                        # (display_name from OAuth) — no custom name, so it can't be faked.
                        attribution_name=None,
                    )
            except Exception as exc:  # best-effort; anonymous is the safe fallback
                _LOG.warning("dashboard: post-login attribution set failed: %s", exc)
        login_session.reset()  # one-time — don't re-show this flow's prompt
        return RedirectResponse("/", status_code=303)

    @app.get("/activity", response_class=HTMLResponse)
    def activity() -> str:
        rows_raw = audit_repo.recent(limit=50)
        rows_html: list[list[str]] = []
        for r in rows_raw:
            occurred = html.escape(r.occurred_at.isoformat()) if r.occurred_at else "—"
            unit = html.escape(r.unit_id or "—")
            action = html.escape(r.action or "—")
            tenant = html.escape(r.tenant_id or "—")
            reason = html.escape(r.reason or "")
            rows_html.append(
                [
                    f'<span class="dim mono">{occurred}</span>',
                    f'<span class="mono">{unit}</span>',
                    action,
                    f'<span class="dim mono">{tenant}</span>',
                    f'<span class="dim">{reason}</span>',
                ]
            )
        table = render_table(
            ["occurred_at", "unit_id", "action", "tenant_id", "reason"],
            rows_html,
            "No assignment activity yet.",
        )
        body = "    <h2>Recent assignment decisions</h2>\n" + table
        return render_page(title="Activity", body=body, active_nav="/activity")

    @app.get("/receipts", response_class=HTMLResponse)
    def receipts() -> str:
        items = results_repo.list_receipts(limit=50)
        rows_html: list[list[str]] = []
        for r in items:
            submitted = (
                html.escape(r.submitted_at.isoformat()) if getattr(r, "submitted_at", None) else "—"
            )
            unit = html.escape(getattr(r, "unit_id", "") or "—")
            receipt_id = html.escape(getattr(r, "receipt_id", "") or "—")
            status = getattr(r, "receipt_status", None) or "—"
            badge_cls = {
                "canonical": "ok",
                "placeholder": "warn",
                "failed": "error",
            }.get(status, "")
            status_html = (
                f'<span class="badge {badge_cls}">{html.escape(status)}</span>'
                if badge_cls
                else html.escape(status)
            )
            rows_html.append(
                [
                    f'<span class="dim mono">{submitted}</span>',
                    f'<span class="mono">{unit}</span>',
                    f'<span class="mono">{receipt_id}</span>',
                    status_html,
                ]
            )
        table = render_table(
            ["submitted_at", "unit_id", "receipt_id", "status"],
            rows_html,
            "No receipts yet. Receipts appear after the worker completes a "
            "work unit and the coordinator's quorum accepts the result.",
        )
        body = (
            "    <h2>Receipts</h2>\n"
            '    <p class="muted">Status legend: <span class="badge ok">canonical</span> '
            "= coordinator issued the COSE-signed receipt; "
            '<span class="badge warn">placeholder</span> = receipt issuance '
            "pending coordinator-side quorum / M7-tail fetch; "
            '<span class="badge error">failed</span> = receipt issuance '
            "failed terminally.</p>\n" + table
        )
        return render_page(title="Receipts", body=body, active_nav="/receipts")

    @app.get("/models", response_class=HTMLResponse)
    def models_page() -> str:
        store = ModelStore(config.models_store_path)
        inv_rows = [
            [
                f'<span class="mono">{html.escape(m.id)}</span>',
                f"{m.size_bytes / 1e9:.2f} GB",
                f'<span class="dim mono">{html.escape(str(m.path))}</span>',
            ]
            for m in store.list()
        ]
        inv_table = render_table(
            ["model id", "size", "path"], inv_rows, "No models in the store yet."
        )
        # Accelerator (drives what this host can run). Live HF browsing is the
        # CLI's job (`model recommend`) — a dashboard page shouldn't query HF on
        # every render.
        acc = detect_accelerator()
        body = (
            "    <h2>This host can run</h2>\n"
            f'    <dl class="kv"><dt>accelerator</dt><dd>{html.escape(acc.label)}</dd>'
            f"<dt>unified memory</dt><dd>{'yes' if acc.unified else 'no'}</dd></dl>\n"
            "    <h2>Local model store (BYOM)</h2>\n"
            f'    <p class="muted">Store: <code>{html.escape(str(store.root))}</code>. '
            "The platform never distributes weights — you host only what you choose to download. "
            "Run <code>auspexai-worker model recommend</code> for HuggingFace models that fit this "
            "host, then <code>model pull &lt;repo&gt; --quant &lt;Q&gt;</code> (or "
            "<code>model setup</code>) to add them.</p>\n" + inv_table
        )
        return render_page(title="Models", body=body, active_nav="/models")

    @app.get("/config", response_class=HTMLResponse)
    def config_page() -> str:
        # Read-only settings, GROUPED so the page reads as sections rather than a
        # flat 15-row dump. Values share one type system: mono only for technical
        # identifiers (url, paths); plain text otherwise — no <code> chips.
        # NB: execute_tenant_code is NOT in these tables — it has its own live
        # "Code-execution policy" section below (these read the frozen daemon-start
        # snapshot, which would disagree after a hot-reload).
        setting_groups: list[tuple[str, list[tuple[str, str, bool]]]] = [
            (
                "Connection",
                [
                    ("coordinator url", html.escape(config.coordinator_url), True),
                    ("heartbeat interval", f"{config.heartbeat_interval_seconds}s", False),
                    ("assignment poll", f"{config.assignment_poll_interval_seconds}s", False),
                ],
            ),
            (
                "Storage",
                [
                    ("state dir", html.escape(str(config.state_dir)), True),
                    ("data dir", html.escape(str(config.data_dir)), True),
                    ("provisioning dir", html.escape(str(config.provisioning_path)), True),
                    ("model store dir", html.escape(str(config.models_store_path)), True),
                ],
            ),
            (
                "Sandbox & safety",
                [
                    ("keystore backend", html.escape(config.keystore_backend or "auto"), False),
                    (
                        "sandbox (bubblewrap)",
                        "yes" if config.sandbox_use_bubblewrap else "no",
                        False,
                    ),
                    (
                        "resource caps",
                        (
                            "off"
                            if not config.sandbox_resource_limits
                            else (
                                f"mem={config.sandbox_memory_max_mb or '∞'}MB · "
                                f"pids={config.sandbox_pids_max or '∞'} (STRICT only)"
                            )
                        ),
                        False,
                    ),
                    ("runner timeout", f"{config.runner_timeout_seconds}s", False),
                ],
            ),
            (
                "Thermal",
                [
                    (
                        "thresholds",
                        f"warn {config.thermal_warn_c:.0f}°C / crit {config.thermal_crit_c:.0f}°C "
                        f"/ resume {config.thermal_resume_c:.0f}°C",
                        False,
                    ),
                ],
            ),
            (
                "Dashboard",
                [
                    (
                        "dashboard",
                        f"{'enabled' if config.dashboard_enabled else 'disabled'} at "
                        f"{config.dashboard_host}:{config.dashboard_port}",
                        False,
                    ),
                    (
                        "upgrade prompt",
                        f"{'enabled' if config.upgrade_prompt_enabled else 'disabled'}"
                        f" (threshold: {config.upgrade_prompt_threshold} units)",
                        False,
                    ),
                ],
            ),
        ]
        # M9 leg 4: the one writable control on this page — the executor-policy
        # setter. Downgrades toward safer modes are one click; enabling
        # provisioned routes through the /executor/confirm deliberate-act step.
        # Buttons + badge reflect the LIVE on-disk policy (the daemon hot-reloads
        # the same value, so there's no stale-snapshot/pending-restart split).
        current = _current_executor_policy(config, config_path)
        setter_buttons: list[str] = []
        if current != "synthetic":
            setter_buttons.append(
                '<form method="post" action="/executor" class="action">'
                '<input type="hidden" name="policy" value="synthetic">'
                '<button type="submit" class="btn">set synthetic (echo only)</button></form>'
            )
        if current != "off":
            setter_buttons.append(
                '<form method="post" action="/executor" class="action">'
                '<input type="hidden" name="policy" value="off">'
                '<button type="submit" class="btn">set off (refuse all)</button></form>'
            )
        if current != "provisioned":
            # the deliberate-act path (confirm step before enabling 3rd-party code);
            # emphasized — it's the consequential consent action on this page.
            setter_buttons.append(
                '<a href="/executor/confirm" class="btn primary" role="button">'
                "enable provisioned…</a>"
            )
        executor_setter = (
            "    <h3>Code-execution policy</h3>\n"
            f"    <p>current: {_executor_indicator(current)}</p>\n"
            f'    <div class="btn-row">{"".join(setter_buttons)}</div>\n'
            '    <p class="muted">Your consent to run third-party tenant code on this '
            "machine. The running daemon <strong>hot-reloads</strong> the change — "
            "effective within one heartbeat, <strong>no restart needed</strong>. "
            "Enabling <strong>provisioned</strong> asks for confirmation first; the "
            "network only routes real (model-gated) experiments to provisioned "
            "workers.</p>\n"
        )
        settings_html = "".join(
            f"    <h3>{title}</h3>\n{render_kv(rows)}\n" for title, rows in setting_groups
        )
        body = (
            "    <h2>Configuration</h2>\n"
            + executor_setter
            + '    <p class="muted">The settings below are read-only — edit '
            "<code>~/.config/auspexai-worker/worker.toml</code> or the matching env var, "
            "then restart the daemon.</p>\n" + settings_html
        )
        return render_page(title="Config", body=body, active_nav="/config")

    # ---- JSON API for future polling refresh ----------------------------

    @app.get("/api/stats")
    def api_stats() -> JSONResponse:
        stats = _gather_stats()
        worker = stats["worker"]
        # Continuous telemetry the baseline poll keeps live: thermal (changes every
        # few seconds as the box works) + work-unit progress. Read fresh each call.
        mon = _thermal_monitor(config)
        thermal_enabled = mon.enabled
        thermal_temp_c: float | None = None
        thermal_state: str | None = None
        if thermal_enabled:
            snap = mon.snapshot()
            thermal_temp_c = (
                round(snap.current_temp_c, 1) if snap.current_temp_c is not None else None
            )
            thermal_state = snap.state.value
        progress = results_repo.progress_summary()
        # Derived volunteer-facing state (§2.1 #11) — kept live by the poll so the
        # status badge flips within a tick on an operator hold / self-pause / etc.
        state_key = state_label = state_tone = state_detail = None
        activity_headline = activity_detail = None
        notice_class = notice_html = ""
        if worker is not None:
            notice_class, notice_html = _update_notice(worker, config)
        if worker is not None:
            now = datetime.now(UTC)
            st = derive_self_state(
                worker,
                thermal_critical=(thermal_enabled and thermal_state == "critical"),
                now=now,
            )
            state_key, state_label, state_tone, state_detail = (
                st.state.value,
                st.label,
                st.tone,
                st.detail,
            )
            recent_submitted = results_repo.recent(limit=1)
            last_submitted_at = recent_submitted[0].submitted_at if recent_submitted else None
            # Plain (headline, detail) for the activity heart — what's HAPPENING,
            # for the heart's own line (the worker-state line is state_label/detail).
            activity_headline, activity_detail = _work_activity(
                config,
                runs_dir=workspace_runs_dir(config.state_dir),
                last_submitted_at=last_submitted_at,
                now=now,
            )
        return JSONResponse(
            {
                "worker_id": worker.worker_id if worker else None,
                "trust_tier": int(worker.trust_tier) if worker else None,
                "worker_state": state_key,
                "state_label": state_label,
                "state_tone": state_tone,
                "state_detail": state_detail,
                "last_heartbeat_at": (
                    worker.last_heartbeat_at.isoformat()
                    if worker and worker.last_heartbeat_at
                    else None
                ),
                "receipts_count": stats["receipts_count"],
                "pending_submissions": stats["pending_submissions"],
                "audit_count": stats["audit_count"],
                "completed_units": progress["completed_units"],
                "distinct_experiments": progress["distinct_experiments"],
                "activity_headline": activity_headline,
                "activity_detail": activity_detail,
                "thermal_enabled": thermal_enabled,
                "thermal_temp_c": thermal_temp_c,
                "thermal_state": thermal_state,
                "coordinator_url": config.coordinator_url,
                # Heart vitals (the worker's operational state): how it computes
                # (accelerator, static — cached at build) + what it runs (execution,
                # the live hot-reloaded policy). thermal + inference are alongside.
                "accelerator": accelerator_label,
                "execution": _current_executor_policy(config, config_path),
                "inference": _inference_status(config),  # live backend reachability (or null)
                # §9 #46: update-available notice (server-built, escaped) + flavor.
                "update_available": bool(notice_html),
                "update_notice_class": notice_class,
                "update_notice_html": notice_html,
                "flavor": config.flavor,
            }
        )

    return app
