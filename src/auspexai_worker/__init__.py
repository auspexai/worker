"""AuspexAI volunteer worker daemon.

Phase 1 (M1): Linux-only worker that generates an Ed25519 keypair, stores it in
the OS keyring (with an encrypted-file fallback), enrolls anonymously (T0) with
the coordinator, and exposes a minimal CLI. See
`Documentation/AuspexAI/v0.1.0/worker_daemon_design.md` and principles doc §5.19.
"""

from importlib.metadata import version as _v

__version__ = _v("auspexai-worker")
