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
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

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


def build_app(*, db: Database, config: WorkerConfig) -> FastAPI:
    """Build the dashboard FastAPI app.

    Args:
        db: opened state DB. Reused across requests; SQLite WAL mode
            and the worker's re-entrant transaction lock make this
            safe alongside the daemon's writer threads.
        config: snapshot of the loaded WorkerConfig. Read-only for
            the dashboard's purposes; config doesn't change inside a
            running daemon.
    """
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

        body = (
            "    <h2>Identity</h2>\n"
            + kv
            + "\n"
            + upgrade_html
            + health_html
            + "\n"
            + progress_html
            + "\n"
            + counts
        )
        return render_page(title="Overview", body=body, active_nav="/")

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
        body = (
            "    <h2>Configuration (read-only)</h2>\n" + render_kv(rows) + "\n"
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
