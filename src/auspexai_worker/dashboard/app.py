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

        rows: list[tuple[str, str, bool]] = [
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
            (
                "last heartbeat",
                html.escape(_fmt_relative(worker.last_heartbeat_at)),
                False,
            ),
            (
                "coordinator",
                f"<code>{html.escape(config.coordinator_url)}</code>",
                False,
            ),
        ]
        kv = render_kv(rows)

        progress = results_repo.progress_summary()
        progress_html = f"""    <h2>Progress</h2>
    <dl class="kv">
      <dt>work units completed</dt><dd>{progress["completed_units"]}</dd>
      <dt>distinct experiments</dt><dd>{progress["distinct_experiments"]}</dd>
    </dl>"""

        upgrade_html = ""
        if (
            worker is not None
            and int(worker.trust_tier) == 0
            and config.upgrade_prompt_enabled
            and progress["completed_units"] >= config.upgrade_prompt_threshold
        ):
            upgrade_html = (
                '    <div class="notice">'
                "You've contributed enough to build a portable track record. "
                "Run <code>auspexai-worker login</code> to claim your contributions."
                "</div>\n"
            )

        counts = f"""    <h2>Activity</h2>
    <dl class="kv">
      <dt>receipts earned</dt><dd>{stats["receipts_count"]}</dd>
      <dt>pending submissions</dt><dd>{stats["pending_submissions"]}</dd>
      <dt>audit-log rows</dt><dd>{stats["audit_count"]}</dd>
      <dt>tenant allow / deny</dt><dd>{stats["tenant_allow_count"]} / {stats["tenant_deny_count"]}</dd>
    </dl>"""

        # Health & execution — what this machine is set to run + its physical state.
        model_count = len(ModelStore(config.models_store_path).list())
        acc = detect_accelerator()
        health_html = f"""    <h2>Health &amp; execution</h2>
    <dl class="kv">
      <dt>tenant code</dt><dd>{_executor_badge(config.execute_tenant_code)}</dd>
      <dt>accelerator</dt><dd>{html.escape(acc.label)}</dd>
      <dt>thermal</dt><dd>{_thermal_html(config)}</dd>
      <dt>models in store</dt><dd>{model_count} (<a href="/models">manage</a>)</dd>
    </dl>"""

        # §2.1 #11 holds: the operator hold (read-only, with the operator's reason)
        # + the volunteer self-pause toggle (the one mutating control on this
        # otherwise read-only dashboard — a low-risk operational lever).
        holds_parts: list[str] = []
        if worker is not None and worker.operator_hold_kind == "pause":
            holds_parts.append(
                '    <div class="notice">Operator hold: <strong>paused</strong> (no-fault) — '
                f"reason: {html.escape(worker.operator_hold_reason or '—')}</div>\n"
            )
        elif worker is not None and worker.operator_hold_kind == "quarantine":
            holds_parts.append(
                '    <div class="notice">Operator hold: <strong>quarantined</strong> — '
                f"reason: {html.escape(worker.operator_hold_reason or '—')}</div>\n"
            )
        if worker is not None and worker.self_paused:
            sp_reason = (
                f" reason: {html.escape(worker.self_pause_reason)}"
                if worker.self_pause_reason
                else ""
            )
            holds_parts.append(
                '    <div class="notice">You self-paused this worker — it stays enrolled '
                f"(tier preserved) but receives no work.{sp_reason} "
                '<form method="post" action="/self-unpause" style="display:inline">'
                '<button type="submit">resume (unpause)</button></form></div>\n'
            )
        else:
            holds_parts.append(
                '    <form method="post" action="/self-pause" style="margin:0.75em 0">'
                '<input type="text" name="reason" placeholder="reason (optional)" '
                'style="padding:0.3em;min-width:18em;margin-right:0.4em"> '
                '<button type="submit">pause this worker</button> '
                '<span class="muted">— stop receiving work; keep enrollment + tier</span>'
                "</form>\n"
            )
        holds_html = "".join(holds_parts)

        body = (
            "    <h2>Identity</h2>\n"
            + kv
            + "\n"
            + holds_html
            + upgrade_html
            + health_html
            + "\n"
            + progress_html
            + "\n"
            + counts
        )
        return render_page(title="Overview", body=body, active_nav="/")

    @app.post("/self-pause")
    async def self_pause(request: Request) -> RedirectResponse:
        """§2.1 #11: the one mutating control on the dashboard — the volunteer's
        own no-fault pause (low-risk; localhost-only). Takes effect within a
        heartbeat (the daemon declares self_paused + stops polling for work).

        The optional `reason` is parsed straight from the urlencoded form body
        (no `python-multipart` dependency) and stored as a local note, mirroring
        the CLI `pause --reason`; empty ⇒ None (no synthetic placeholder)."""
        raw = (await request.body()).decode("utf-8", "replace")
        reason = (parse_qs(raw).get("reason", [""])[0] or "").strip() or None
        if self_repo.get() is not None:
            self_repo.set_self_pause(True, reason=reason)
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
    # a safer mode is one click. The write applies on the next daemon restart
    # (same as `executor set`). Tier-agnostic by design — owner consent is a
    # different axis than the network's trust tier (see M9 leg-4 design).
    @app.get("/executor/confirm", response_class=HTMLResponse)
    def executor_confirm() -> str:
        body = (
            "    <h2>Enable provisioned execution?</h2>\n"
            '    <div class="notice">Switching to <strong>provisioned</strong> means '
            "this machine will run <strong>third-party tenant code</strong> — but only "
            "executors that were operator-staged locally and whose hash matches the "
            "coordinator's manifest (anything else is refused, never echoed). This is "
            "your consent, on your hardware. The change applies on the next daemon "
            "restart.</div>\n"
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
            (
                "execute tenant code",
                _executor_badge(config.execute_tenant_code),
                False,
            ),
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
        current = config.execute_tenant_code
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
            "machine. Applies on the next daemon restart. Enabling "
            "<strong>provisioned</strong> requires confirmation; the network only "
            "routes real (model-gated) experiments to provisioned workers.</p>\n"
        )
        body = (
            "    <h2>Configuration</h2>\n"
            + executor_setter
            + "    <h3>Other settings (read-only)</h3>\n"
            + render_kv(rows)
            + "\n"
            '    <p class="muted">Edit '
            "<code>~/.config/auspexai-worker/worker.toml</code> or set the "
            "matching env var to change a value, then restart the daemon. "
            "The dashboard reflects the loaded values, not the current "
            "file contents.</p>"
        )
        return render_page(title="Config", body=body, active_nav="/config")

    # ---- JSON API for future polling refresh ----------------------------

    @app.get("/api/stats")
    def api_stats() -> JSONResponse:
        stats = _gather_stats()
        worker = stats["worker"]
        return JSONResponse(
            {
                "worker_id": worker.worker_id if worker else None,
                "trust_tier": int(worker.trust_tier) if worker else None,
                "last_heartbeat_at": (
                    worker.last_heartbeat_at.isoformat()
                    if worker and worker.last_heartbeat_at
                    else None
                ),
                "receipts_count": stats["receipts_count"],
                "pending_submissions": stats["pending_submissions"],
                "audit_count": stats["audit_count"],
                "coordinator_url": config.coordinator_url,
            }
        )

    return app
