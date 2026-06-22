"""HTML templates for the worker dashboard.

Plain Python f-strings; no Jinja2 or other template engine dependency.
All HTML is escape-safe at insertion points via `html.escape()` in the
route handlers; the layout itself is trusted (we wrote it).
"""

from __future__ import annotations

_BASE_CSS = """
* { box-sizing: border-box; }
html { font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif; line-height: 1.5; color: #d4d4dc; background: #0a0e1a; }
body { max-width: 1024px; margin: 0 auto; padding: 1.5em 1em 3em; }
header { display: flex; align-items: baseline; justify-content: space-between; border-bottom: 1px solid #2a2e3a; padding-bottom: 0.75em; margin-bottom: 1.5em; }
header h1 { margin: 0; font-size: 1.3em; font-weight: 600; color: #ffffff; }
header h1 .brand { color: #A78BFA; }
nav { display: flex; gap: 1em; font-size: 0.95em; }
nav a { color: #9ca3af; text-decoration: none; padding: 0.2em 0; border-bottom: 2px solid transparent; }
nav a:hover { color: #d4d4dc; }
nav a.active { color: #ffffff; border-bottom-color: #A78BFA; }
h2 { font-size: 1.05em; font-weight: 600; margin: 1.5em 0 0.5em; color: #ffffff; }
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
th, td { text-align: left; padding: 0.45em 0.6em; border-bottom: 1px solid #2a2e3a; }
th { color: #9ca3af; font-weight: 500; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.85em; }
.dim { color: #9ca3af; }
.kv { display: grid; grid-template-columns: 12em 1fr; gap: 0.3em 1em; }
.kv dt { color: #9ca3af; }
.kv dd { margin: 0; }
.kv dd.mono { word-break: break-all; }
/* card grid — matches the researcher dashboard's .grid/.field metric cards */
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.6em; margin: 0.5em 0 1.3em; }
.field { border: 1px solid #1e2638; border-radius: 6px; padding: 0.55em 0.75em; background: #0e1424; display: flex; flex-direction: column; gap: 0.2em; min-width: 0; }
.field .k { color: #7c849a; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.04em; }
.field .v { color: #e6ebf5; }
.field .v.mono { font-family: ui-monospace, monospace; font-size: 0.85em; word-break: break-all; }
/* compact identity line in the heart header (worker_id · version) */
.heart .heart-id { font-size: 0.72em; color: #7c849a; font-family: ui-monospace, monospace; margin: -0.2em 0 0.1em; }
.badge { display: inline-block; padding: 0.1em 0.55em; border-radius: 3px; font-size: 0.8em; font-weight: 500; }
.badge.tier-0 { background: #2a2e3a; color: #9ca3af; }
.badge.tier-1, .badge.tier-2, .badge.tier-3, .badge.tier-4 { background: #312e81; color: #c4b5fd; }
.badge.ok { background: #14532d; color: #86efac; }
.badge.warn { background: #713f12; color: #fcd34d; }
.badge.error { background: #7f1d1d; color: #fca5a5; }
.meta { color: #6b7280; font-size: 0.85em; margin-top: 2em; padding-top: 1em; border-top: 1px solid #2a2e3a; }
.muted { color: #6b7280; }
code { font-family: ui-monospace, monospace; background: #1a1e2a; padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.85em; }
.empty { color: #6b7280; font-style: italic; padding: 1em 0; }
.notice { background: #1e3a5f; border: 1px solid #3b82f6; border-radius: 6px; padding: 0.75em 1em; margin: 1em 0; color: #93c5fd; }
.notice.ok { background: #0f2a1a; border-color: #14532d; color: #86efac; }
.notice.fault { background: #3a1e1e; border-color: #b91c1c; color: #fca5a5; }
.notice code { background: #0a0e1a; }
.notice .copy-cmd { background: #1f2937; border: 1px solid #3b82f6; color: inherit; border-radius: 4px; padding: 0.15em 0.6em; margin-left: 0.4em; font: inherit; font-size: 0.85em; cursor: pointer; }
.notice .copy-cmd:hover { background: #2a2e3a; }
.live-ind { font-size: 0.6em; font-weight: 500; color: #86efac; margin-left: 0.5em; vertical-align: middle; }
/* the state banner is a HOLD alert now (option B): empty for active/idle → collapse it */
[data-live="state_banner"]:empty { display: none; }
/* activity heart (overview) — shared identity: cyan=working, blue=idle, red=problem */
.heart { border: 1px solid #1f6b78; border-radius: 12px; background: linear-gradient(180deg,#101727 0%,#0c1322 100%); padding: 1rem 1.1rem; margin: 1em 0; display: flex; flex-direction: column; gap: 0.6rem; }
.heart header { display: flex; align-items: center; gap: 0.55rem; margin: 0; }
.heart .heart-h { margin: 0; font-size: 0.95rem; border: none; padding: 0; }
.heart .heart-status { margin-left: auto; font-size: 0.72rem; color: #8b93a7; text-transform: lowercase; }
.pulse-dot { width: 10px; height: 10px; border-radius: 50%; background: #2a3450; flex: none; }
.pulse-dot.working { background: #67e8f9; animation: heartbeat 1.1s ease-out infinite; }
.pulse-dot.idle { background: #4a7dff; }
.pulse-dot.problem { background: #fca5a5; }
@keyframes heartbeat { 0% { box-shadow: 0 0 0 0 rgba(103,232,249,0.55); } 70% { box-shadow: 0 0 0 8px rgba(103,232,249,0); } 100% { box-shadow: 0 0 0 0 rgba(103,232,249,0); } }
.heart .strip { display: flex; align-items: flex-end; gap: 2px; height: 64px; padding: 4px; background: #080d18; border: 1px solid #161d2c; border-radius: 8px; overflow: hidden; }
.heart .bar { flex: 0 0 3px; min-width: 3px; background: #233049; border-radius: 2px; align-self: flex-end; }
.heart .bar.beat { background: #67e8f9; }
.heart .strip-empty { margin: auto; color: #5b6478; font-size: 0.8rem; }
.heart .narration { margin: 0; font-size: 0.86rem; color: #b8bfd0; }
.heart .narration.good { color: #67e8f9; }
.heart .narration.reassure { color: #4a7dff; }
.heart .narration.bad { color: #fca5a5; }
.heart .heart-vitals { display: flex; flex-wrap: wrap; gap: 0.9rem; font-size: 0.76rem; color: #9aa3b8; }
.heart .vital { display: inline-flex; align-items: center; gap: 0.35rem; }
.heart .vital.muted { color: #6b7488; }
.heart .vital.warn { color: #fbbf24; }
.heart .vital.bad { color: #fca5a5; }
.heart .vdot { width: 7px; height: 7px; border-radius: 50%; background: #2a3450; display: inline-block; }
.heart .vdot.ok { background: #6ee7b7; }
.heart .vdot.down { background: #fca5a5; }
.heart .heart-metrics { display: flex; gap: 1.3rem; }
.heart .hm { display: flex; flex-direction: column; gap: 0.1rem; }
.heart .hm .n { font-size: 1.05rem; font-weight: 600; color: #e6ebf5; }
.heart .hm .l { font-size: 0.66rem; color: #7c849a; text-transform: uppercase; letter-spacing: 0.04em; }
"""

# Baseline-poll live updater (M6 #3, worker side). Completes the same principle
# the consoles use — "poll is the truth" — on the worker dashboard: a small
# vanilla-JS loop re-reads /api/stats and refreshes any element tagged
# data-live="<key>", flipping the header indicator to "stale" if the poll fails.
# No SSE doorbell here (the worker daemon has no event bus yet; that's a flagged
# follow-on) — but a 10s poll on a localhost page is indistinguishable from live.
_LIVE_SCRIPT = """  <script>
  (function () {
    var ind = document.getElementById('live-ind');
    function rel(iso) {
      if (!iso) return 'never';
      var s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
      if (s < 0) return new Date(iso).toLocaleString();
      if (s < 60) return s + 's ago';
      if (s < 3600) return Math.floor(s / 60) + 'm ago';
      if (s < 86400) return Math.floor(s / 3600) + 'h ago';
      return Math.floor(s / 86400) + 'd ago';
    }
    function setLive(ok) {
      if (!ind) return;
      ind.textContent = ok ? '\\u25CF live' : '\\u25CF stale';
      ind.style.color = ok ? '#86efac' : '#fbbf24';
      ind.title = ok ? 'live \\u2014 this page auto-updates (poll); no refresh needed'
                     : 'stale \\u2014 auto-refresh is failing right now';
    }
    // ── Activity heart (overview only). Pulse = this worker's completed units
    // over time, accumulated client-side from the same poll. Same color identity
    // as the researcher/operator hearts: cyan=working, blue=idle, red=problem.
    var heartHist = [];
    function heartState(d) {
      if (!d.worker_id) return 'idle';
      if (d.state_tone === 'warn' || d.state_tone === 'error') return 'problem';
      if (d.thermal_enabled && d.thermal_state === 'critical') return 'problem';
      var h = String(d.activity_headline || '').toLowerCase();
      if (h.indexOf('running') >= 0 || h.indexOf('receiving') >= 0) return 'working';
      return 'idle';
    }
    function renderHeart(d) {
      var strip = document.getElementById('heart-strip');
      if (!strip) return; // not the overview page
      heartHist.push({ t: Date.now(), c: d.completed_units || 0 });
      if (heartHist.length > 80) heartHist.shift();
      var beats = [], maxD = 1;
      for (var i = 1; i < heartHist.length; i++) {
        var dlt = Math.max(0, heartHist[i].c - heartHist[i - 1].c);
        beats.push(dlt);
        if (dlt > maxD) maxD = dlt;
      }
      if (beats.length === 0) {
        strip.innerHTML = '<span class="strip-empty">listening\\u2026</span>';
      } else {
        var bars = '';
        for (var j = 0; j < beats.length; j++) {
          var hgt = beats[j] > 0 ? (16 + Math.round((beats[j] / maxD) * 44)) : 2;
          bars += '<span class="bar' + (beats[j] > 0 ? ' beat' : '') + '" style="height:' + hgt + 'px"></span>';
        }
        strip.innerHTML = bars;
      }
      var st = heartState(d);
      var dot = document.getElementById('heart-dot');
      if (dot) dot.className = 'pulse-dot ' + st;
      var statusEl = document.getElementById('heart-status');
      if (statusEl) statusEl.textContent = st === 'working' ? 'working' : (st === 'problem' ? 'attention' : 'idle');
      var narr = document.getElementById('heart-narration');
      if (narr) {
        // Activity/state only — the unit/experiment counts are the metrics row
        // below, so the line speaks to what's HAPPENING, not the totals.
        var parts = [];
        if (st === 'problem' && d.state_label) {
          // a hold (paused/quarantined/overheating) — say so, don't claim "idle";
          // the loud detail + reason is in the banner above.
          parts.push(String(d.state_label).toLowerCase());
        } else if (d.activity_headline) {
          parts.push(String(d.activity_detail ? (d.activity_headline + ' \\u2014 ' + d.activity_detail) : d.activity_headline).toLowerCase());
        }
        narr.textContent = parts.join(' \\u00B7 ') || 'waiting for work\\u2026';
        narr.className = 'narration ' + (st === 'working' ? 'good' : (st === 'problem' ? 'bad' : 'reassure'));
      }
      var vit = document.getElementById('heart-vitals');
      if (vit) {
        var v = [];
        // coordinator connection = heartbeat freshness (the worker→coordinator link);
        // the URL rides the tooltip. flavor is a static fact (Identity owns it).
        var hbMs = d.last_heartbeat_at ? (Date.now() - new Date(d.last_heartbeat_at).getTime()) : null;
        var coordOk = hbMs != null && hbMs < 180000;
        var coordTxt = !d.worker_id ? 'coordinator \\u00B7 not enrolled'
          : (hbMs == null ? 'coordinator \\u00B7 no contact'
          : (coordOk ? 'coordinator \\u00B7 connected ' + rel(d.last_heartbeat_at)
          : 'coordinator \\u00B7 no contact ' + rel(d.last_heartbeat_at)));
        var coordUrl = String(d.coordinator_url || '').replace(/[<>&"]/g, '');
        v.push('<span class="vital' + (coordOk ? '' : ' bad') + '" title="' + coordUrl + '"><i class="vdot ' + (coordOk ? 'ok' : 'down') + '"></i>' + coordTxt + '</span>');
        if (d.thermal_enabled && d.thermal_state) {
          var tcls = d.thermal_state === 'critical' ? 'bad' : (d.thermal_state === 'warm' ? 'warn' : '');
          v.push('<span class="vital ' + tcls + '">' + (d.thermal_temp_c != null ? d.thermal_temp_c + '\\u00B0C' : String(d.thermal_state).replace(/[<>&]/g, '')) + '</span>');
        }
        if (d.inference) {
          var inf = d.inference;
          var infName = String(inf.backend || 'inference').replace(/[<>&]/g, '');
          var infVer = inf.version ? ' v' + String(inf.version).replace(/[<>&]/g, '') : '';
          var infTxt = infName + infVer + (inf.reachable ? '' : ' \\u2014 unreachable');
          v.push('<span class="vital' + (inf.reachable ? '' : ' bad') + '"><i class="vdot ' + (inf.reachable ? 'ok' : 'down') + '"></i>' + infTxt + '</span>');
        }
        vit.innerHTML = v.join('');
      }
      var hu = document.getElementById('heart-units'); if (hu) hu.textContent = d.completed_units || 0;
      var he = document.getElementById('heart-exps'); if (he) he.textContent = d.distinct_experiments || 0;
    }
    function tick() {
      fetch('/api/stats', { cache: 'no-store' })
        .then(function (r) { if (!r.ok) throw new Error('http ' + r.status); return r.json(); })
        .then(function (d) {
          renderHeart(d);
          document.querySelectorAll('[data-live]').forEach(function (el) {
            var k = el.getAttribute('data-live');
            if (k === 'thermal') {
              if (!d.thermal_enabled) {
                el.innerHTML = '<span class="muted">no thermal sensor \\u2014 governor inactive on this host</span>';
              } else {
                var cls = { ok: 'ok', warm: 'warn', critical: 'error' }[d.thermal_state] || '';
                var temp = (d.thermal_temp_c != null) ? (d.thermal_temp_c + '\\u00B0C') : '\\u2014';
                var st = String(d.thermal_state == null ? '' : d.thermal_state).replace(/[<>&]/g, '');
                el.innerHTML = temp + ' <span class="badge ' + cls + '">' + st + '</span>';
              }
              return;
            }
            if (k === 'worker_state') {
              if (d.state_label != null) el.textContent = d.state_label;
              el.className = 'badge ' + (d.state_tone || '');
              return;
            }
            if (k === 'state_banner') {
              // The dynamic state banner: server-built (already escaped) inner
              // HTML + class, so "receiving work" vs "idle" flips live.
              if (d.state_banner_html != null) el.innerHTML = d.state_banner_html;
              if (d.state_banner_class != null) el.className = d.state_banner_class;
              return;
            }
            if (k === 'update_notice') {
              // §9 #46 update-available notice: server-built (escaped) inner
              // HTML + class, so a release announcement appears without a
              // page reload (and disappears after an upgrade).
              if (d.update_notice_html != null) el.innerHTML = d.update_notice_html;
              if (d.update_notice_class != null) el.className = d.update_notice_class;
              return;
            }
            if (!(k in d)) return;
            el.textContent = (k === 'last_heartbeat_at') ? rel(d[k]) : d[k];
          });
          setLive(true);
        })
        .catch(function () { setLive(false); });
    }
    tick();
    setInterval(tick, 10000);
  })();
  </script>
"""

_NAV_ITEMS = [
    ("/", "Overview"),
    ("/activity", "Activity"),
    ("/models", "Models"),
    ("/receipts", "Receipts"),
    ("/citation", "Citation"),
    ("/config", "Config"),
]


def render_page(*, title: str, body: str, active_nav: str, live: bool = False) -> str:
    """Wrap a body fragment in the base layout. `active_nav` is the
    path of the current page so we can highlight the right nav item.

    `live=True` adds the header "● live" indicator + the baseline-poll script
    (the overview passes it; static log/config pages don't need it)."""
    nav_html_parts = []
    for path, label in _NAV_ITEMS:
        cls = "active" if path == active_nav else ""
        nav_html_parts.append(f'<a href="{path}" class="{cls}">{label}</a>')
    nav_html = "\n      ".join(nav_html_parts)

    live_ind = (
        ' <span id="live-ind" class="live-ind"'
        ' title="live — this page auto-updates (poll); no refresh needed">● live</span>'
        if live
        else ""
    )
    live_script = _LIVE_SCRIPT if live else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title} — auspex[ai] worker</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>{_BASE_CSS}</style>
</head>
<body>
  <header>
    <h1><span class="brand">auspex[ai]</span> worker{live_ind}</h1>
    <nav>
      {nav_html}
    </nav>
  </header>
  <main>
{body}
  </main>
  <p class="meta">Local volunteer dashboard · localhost-only · read-only.
    Withdrawal and tier upgrades remain CLI-only.
    See <a href="https://github.com/auspexai/worker" style="color:#A78BFA">github.com/auspexai/worker</a>.</p>
{live_script}</body>
</html>"""


def render_kv(rows: list[tuple[str, str, bool]]) -> str:
    """Render a (label, value, is_mono) list as a definition list."""
    parts = ['    <dl class="kv">']
    for label, value, mono in rows:
        cls = "mono" if mono else ""
        parts.append(f"      <dt>{label}</dt>")
        parts.append(f'      <dd class="{cls}">{value}</dd>')
    parts.append("    </dl>")
    return "\n".join(parts)


def render_cards(rows: list[tuple[str, str, bool]]) -> str:
    """Render a (label, value, is_mono) list as a responsive grid of cards —
    the same .grid/.field aesthetic as the researcher dashboard's experiment
    page. Value html is inserted verbatim (caller escapes)."""
    parts = ['    <div class="grid">']
    for label, value, mono in rows:
        v_cls = "v mono" if mono else "v"
        parts.append(
            f'      <div class="field"><span class="k">{label}</span>'
            f'<span class="{v_cls}">{value}</span></div>'
        )
    parts.append("    </div>")
    return "\n".join(parts)


def render_table(headers: list[str], rows: list[list[str]], empty_msg: str) -> str:
    """Render a simple HTML table. Each row's cells are inserted verbatim
    (caller is responsible for html.escape). Caller can embed
    <span class="mono"> / <span class="dim"> inside cells."""
    if not rows:
        return f'    <p class="empty">{empty_msg}</p>'
    parts = ["    <table>", "      <thead><tr>"]
    for h in headers:
        parts.append(f"        <th>{h}</th>")
    parts.append("      </tr></thead>")
    parts.append("      <tbody>")
    for row in rows:
        parts.append("        <tr>")
        for cell in row:
            parts.append(f"          <td>{cell}</td>")
        parts.append("        </tr>")
    parts.append("      </tbody>")
    parts.append("    </table>")
    return "\n".join(parts)
