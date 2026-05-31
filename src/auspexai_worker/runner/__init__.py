"""Runner subprocess — the untrusted half of the §5.17 two-tier model.

The runner is invoked as a separate process (typically via bubblewrap from
the daemon) per work unit. It is short-lived, sandboxed, and never holds
the worker keystore. It reads a work-unit envelope from stdin, executes
the tenant's payload, and writes a Result body to the path in
$AUSPEXAI_OUTPUT_PATH. The daemon signs and submits the Result; the
runner has no network access in production sandbox config.

M4 ships a *synthetic* executor that echoes the input payload back as
output. Tenant code (the first real tenant Vigiles, eventually) replaces
the synthetic executor in a later milestone per §5.3.
"""

from __future__ import annotations

from .main import main

__all__ = ["main"]
