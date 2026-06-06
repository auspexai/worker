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
.notice code { background: #0a0e1a; }
"""

_NAV_ITEMS = [
    ("/", "Overview"),
    ("/activity", "Activity"),
    ("/models", "Models"),
    ("/receipts", "Receipts"),
    ("/config", "Config"),
]


def render_page(*, title: str, body: str, active_nav: str) -> str:
    """Wrap a body fragment in the base layout. `active_nav` is the
    path of the current page so we can highlight the right nav item."""
    nav_html_parts = []
    for path, label in _NAV_ITEMS:
        cls = "active" if path == active_nav else ""
        nav_html_parts.append(f'<a href="{path}" class="{cls}">{label}</a>')
    nav_html = "\n      ".join(nav_html_parts)

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
    <h1><span class="brand">auspex[ai]</span> worker</h1>
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
</body>
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
