"""FastAPI app factory for the worker dashboard.

Four read-only HTML pages plus a small JSON `/api/stats` endpoint
for live numbers (could feed a future polling refresh in the UI; for
now the pages render server-side on each request).

The app captures the worker's local SQLite state DB + the WorkerConfig
at construction time and reads from them on each request. No
write-side surface; the daemon's existing threads remain the only
writers.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
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
from auspexai_worker.worker_state import SelfState, derive_self_state
from auspexai_worker.workspace import workspace_runs_dir

from .templates import render_kv, render_page, render_table


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


def _executor_badge(policy: str) -> str:
    """Make the code-execution consent setting unmissable (§5.14)."""
    label, cls = {
        "synthetic": ("synthetic only — no third-party code", "ok"),
        "provisioned": ("runs provisioned tenant code", "warn"),
        "off": ("off — refuses all work", ""),
    }.get(policy, (policy, ""))
    return f'<span class="badge {cls}">{html.escape(label)}</span>'


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


def _state_banner(
    worker,
    state,
    config: WorkerConfig,
    *,
    runs_dir: Path,
    last_submitted_at: datetime | None,
    now: datetime,
) -> tuple[str, str]:
    """Return (css_class, inner_html) for the overview state banner. Shared by
    the server render and /api/stats so the banner updates live + identically.
    ACTIVE uses the dynamic work-activity headline (accurate, never
    overclaiming); every hold state uses its own already-accurate label+detail."""
    if state.state is SelfState.ACTIVE:
        headline, detail = _work_activity(
            config, runs_dir=runs_dir, last_submitted_at=last_submitted_at, now=now
        )
        cls = "notice ok"
    else:
        headline, detail = state.label, state.detail
        cls = "notice fault" if state.fault else "notice"
    inner = f"<strong>{html.escape(headline)}</strong> — {html.escape(detail)}"
    return cls, inner


def _inference_html(config: WorkerConfig) -> str:
    """W-S: an extra Health & execution row when the worker is configured to
    serve models for inference tenants. Returns '' (no row) when
    `[inference] backend = "none"` — absent means 'not an inference host',
    mirroring the heartbeat wire. When ollama, probes the backend directly so
    the volunteer sees whether it's actually reachable."""
    backend = getattr(config, "inference_backend", "none")
    if backend == "none":
        return ""
    if backend == "ollama":
        try:
            from auspexai_worker.inference import OllamaBackend

            healthy = OllamaBackend(config.inference_ollama_url).is_healthy()
        except Exception:
            healthy = False
        badge = (
            '<span class="badge ok">reachable</span>'
            if healthy
            else '<span class="badge error">unreachable — start Ollama</span>'
        )
        url = html.escape(config.inference_ollama_url)
        return f"\n      <dt>inference backend</dt><dd>ollama @ <code>{url}</code> {badge}</dd>"
    return f"\n      <dt>inference backend</dt><dd>{html.escape(str(backend))}</dd>"


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
    app = FastAPI(
        title="AuspexAI Worker — local dashboard",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    self_repo = WorkerSelfRepository(db)
    audit_repo = AssignmentAuditRepository(db)
    results_repo = SubmittedResultRepository(db)
    pending_repo = PendingSubmissionRepository(db)
    tenant_lists = TenantListRepository(db)

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

        state = derive_self_state(
            worker, thermal_critical=_thermal_critical(config), now=datetime.now(UTC)
        )

        # ── Status: every live "right now" signal in ONE place (the state itself
        # is the banner above; here are the supporting live signals the volunteer
        # asked to have grouped — heartbeat, thermal, what's executing, serving).
        # Current temp/thermal lives here (status); the thresholds are a setting,
        # on Config. Executor mode is read-only here (change it on Config).
        model_count = len(ModelStore(config.models_store_path).list())
        acc = detect_accelerator()
        executor_cell = (
            f"{_executor_badge(_current_executor_policy(config, config_path))} "
            '<a href="/config" class="dim">change</a>'
        )
        # W-S: the inference-backend reachability row (only when configured).
        inference_row = _inference_html(config)
        status_html = f"""    <h2>Status</h2>
    <dl class="kv">
      <dt>last heartbeat</dt><dd><span data-live="last_heartbeat_at">{html.escape(_fmt_relative(worker.last_heartbeat_at))}</span></dd>
      <dt>thermal</dt><dd data-live="thermal">{_thermal_html(config)}</dd>
      <dt>accelerator</dt><dd>{html.escape(acc.label)}</dd>
      <dt>executor mode</dt><dd>{executor_cell}</dd>
      <dt>models in store</dt><dd>{model_count} (<a href="/models">manage</a>)</dd>{inference_row}
    </dl>"""

        # ── Contribution: what this worker has done (Progress + Activity merged —
        # they answered the same question in two boxes).
        progress = results_repo.progress_summary()
        contribution_html = f"""    <h2>Contribution</h2>
    <dl class="kv">
      <dt>work units completed</dt><dd><span data-live="completed_units">{progress["completed_units"]}</span></dd>
      <dt>distinct experiments</dt><dd><span data-live="distinct_experiments">{progress["distinct_experiments"]}</span></dd>
      <dt>receipts earned</dt><dd><span data-live="receipts_count">{stats["receipts_count"]}</span></dd>
      <dt>pending submissions</dt><dd><span data-live="pending_submissions">{stats["pending_submissions"]}</span></dd>
      <dt>audit-log rows</dt><dd><span data-live="audit_count">{stats["audit_count"]}</span></dd>
      <dt>tenant allow / deny</dt><dd>{stats["tenant_allow_count"]} / {stats["tenant_deny_count"]}</dd>
    </dl>"""

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

        # ── Identity: the static "who is this worker" facts.
        identity_rows: list[tuple[str, str, bool]] = [
            ("worker_id", html.escape(worker.worker_id), True),
            ("worker version", f"<code>{html.escape(__version__)}</code>", False),
            ("trust tier", _tier_badge(int(worker.trust_tier)), False),
            ("public key", html.escape(worker.pubkey_hex), True),
            (
                "enrolled",
                html.escape(worker.enrolled_at.isoformat())
                + f' <span class="muted">({_fmt_relative(worker.enrolled_at)})</span>',
                False,
            ),
            ("coordinator", f"<code>{html.escape(config.coordinator_url)}</code>", False),
        ]
        identity_html = "    <h2>Identity</h2>\n" + render_kv(identity_rows)

        # §2.1 #11 (volunteer surface) + I4 (ui_triage_first_ia_redesign.md §5):
        # the state banner leads the page so "is anything wrong with my box?" is
        # answered first — ALWAYS rendered. For an ACTIVE worker the second
        # clause is DYNAMIC + accurate: it claims "receiving work" only when work
        # is actually flowing (a live runner / a recently-submitted unit), else
        # it says idle/available. Hold states use their own accurate detail
        # (quarantine = fault → red; every other hold is no-fault → neutral).
        recent_submitted = results_repo.recent(limit=1)
        last_submitted_at = recent_submitted[0].submitted_at if recent_submitted else None
        banner_cls, banner_inner = _state_banner(
            worker,
            state,
            config,
            runs_dir=workspace_runs_dir(config.state_dir),
            last_submitted_at=last_submitted_at,
            now=datetime.now(UTC),
        )
        state_banner = (
            f'    <div class="{banner_cls}" data-live="state_banner">{banner_inner}</div>\n'
        )

        # The self-pause control is an ACTION (not status) — offered only where
        # the volunteer can actually act. An operator hold (pause OR quarantine)
        # is operator-controlled — don't dangle a pause/unpause that wouldn't
        # change the work state; if self-paused *under* an operator hold, say
        # resuming won't restore work until the hold lifts.
        pause_control = ""
        if worker.self_paused:
            note = ""
            if worker.operator_hold_kind is not None:
                note = (
                    ' <span class="muted">(an operator hold is also active — resuming '
                    "won't restore work until the operator lifts it)</span>"
                )
            pause_control = (
                '    <form method="post" action="/self-unpause" style="margin:0.75em 0">'
                '<button type="submit">resume (unpause)</button>' + note + "</form>\n"
            )
        elif worker.operator_hold_kind is None:
            pause_control = (
                '    <form method="post" action="/self-pause" style="margin:0.75em 0">'
                '<button type="submit">pause this worker</button> '
                '<span class="muted">— stop receiving work; keep enrollment + tier</span>'
                "</form>\n"
            )

        body = (
            state_banner
            + pause_control
            + status_html
            + "\n"
            + contribution_html
            + "\n"
            + upgrade_html
            + identity_html
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
        rows: list[tuple[str, str, bool]] = [
            (
                "coordinator_url",
                f"<code>{html.escape(config.coordinator_url)}</code>",
                False,
            ),
            (
                "heartbeat interval",
                f"{config.heartbeat_interval_seconds}s",
                False,
            ),
            (
                "assignment poll interval",
                f"{config.assignment_poll_interval_seconds}s",
                False,
            ),
            ("state dir", html.escape(str(config.state_dir)), True),
            ("data dir", html.escape(str(config.data_dir)), True),
            (
                "keystore backend",
                html.escape(config.keystore_backend or "auto"),
                False,
            ),
            (
                "sandbox use_bubblewrap",
                "yes" if config.sandbox_use_bubblewrap else "no",
                False,
            ),
            (
                "runner timeout",
                f"{config.runner_timeout_seconds}s",
                False,
            ),
            # NB: execute_tenant_code is NOT listed here — it has its own live
            # "Code-execution policy" section above (this table reads the frozen
            # daemon-start snapshot, which would disagree after a hot-reload).
            ("provisioning dir", html.escape(str(config.provisioning_path)), True),
            ("model store dir", html.escape(str(config.models_store_path)), True),
            (
                "thermal thresholds",
                f"warn {config.thermal_warn_c:.0f}°C / crit {config.thermal_crit_c:.0f}°C "
                f"/ resume {config.thermal_resume_c:.0f}°C",
                False,
            ),
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
                '<form method="post" action="/executor" style="display:inline">'
                '<input type="hidden" name="policy" value="synthetic">'
                '<button type="submit">set synthetic (echo only)</button></form>'
            )
        if current != "off":
            setter_buttons.append(
                '<form method="post" action="/executor" style="display:inline;margin-left:0.5em">'
                '<input type="hidden" name="policy" value="off">'
                '<button type="submit">set off (refuse all)</button></form>'
            )
        if current != "provisioned":
            # the deliberate-act path (confirm step before enabling 3rd-party code)
            setter_buttons.append(
                '<a href="/executor/confirm" style="margin-left:0.5em" '
                'role="button">enable provisioned…</a>'
            )
        executor_setter = (
            "    <h3>Code-execution policy</h3>\n"
            f"    <p>current: {_executor_badge(current)}</p>\n"
            f'    <div style="margin:0.5em 0">{"".join(setter_buttons)}</div>\n'
            '    <p class="muted">Your consent to run third-party tenant code on this '
            "machine. The running daemon <strong>hot-reloads</strong> the change — "
            "effective within one heartbeat, <strong>no restart needed</strong>. "
            "Enabling <strong>provisioned</strong> asks for confirmation first; the "
            "network only routes real (model-gated) experiments to provisioned "
            "workers.</p>\n"
        )
        body = (
            "    <h2>Configuration</h2>\n"
            + executor_setter
            + "    <h3>Other settings (read-only)</h3>\n"
            + render_kv(rows)
            + "\n"
            '    <p class="muted">The code-execution policy above is live-editable. '
            "Other settings: edit <code>~/.config/auspexai-worker/worker.toml</code> or "
            "the matching env var, then restart the daemon.</p>"
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
        state_key = state_label = state_tone = None
        banner_class = banner_html = None
        if worker is not None:
            now = datetime.now(UTC)
            st = derive_self_state(
                worker,
                thermal_critical=(thermal_enabled and thermal_state == "critical"),
                now=now,
            )
            state_key, state_label, state_tone = st.state.value, st.label, st.tone
            # The dynamic state banner (kept live so "receiving work" vs "idle"
            # flips within a tick as work starts/stops) — same helper the page
            # render uses, so server + poll agree exactly.
            recent_submitted = results_repo.recent(limit=1)
            last_submitted_at = recent_submitted[0].submitted_at if recent_submitted else None
            banner_class, banner_html = _state_banner(
                worker,
                st,
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
                "state_banner_class": banner_class,
                "state_banner_html": banner_html,
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
                "thermal_enabled": thermal_enabled,
                "thermal_temp_c": thermal_temp_c,
                "thermal_state": thermal_state,
                "coordinator_url": config.coordinator_url,
            }
        )

    return app
