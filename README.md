# AuspexAI Worker

The volunteer worker process for [AuspexAI](https://github.com/auspexai) — runs on volunteer machines, executes work units dispatched by the coordinator, donates compute to AI safety research.

## Status

**Phase 1 — M1 + M2 SHIPPED 2026-05-20.** First-run enrollment and signed-heartbeat loop both verified end-to-end against the coordinator. Phase 1 worker target is Linux x86_64 + ARM64 (per principles doc §5.13 + §5.19); macOS/WSL2 packaging arrives in Phase 2.

What's live:

- `auspexai-worker bootstrap` — generates an Ed25519 keypair, persists to the OS keyring (libsecret) or an encrypted-file fallback, enrolls anonymously (T0) with the coordinator via `POST /api/v0/workers/enroll`, records the assigned `worker_id` in the local SQLite state DB. Idempotent on re-run.
- `auspexai-worker status` — shows worker_id, tier, pubkey fingerprint, enrollment timestamp, last heartbeat.
- `auspexai-worker daemon` — runs the heartbeat loop: every `heartbeat_interval_seconds` posts a signed `POST /api/v0/workers/{id}/heartbeat` with refreshed capabilities (OS, arch, RAM total, CPU count, GPU presence) plus any declared resource caps from `[resources]`. SIGTERM/SIGINT trigger a clean stop. `--max-ticks=N` for debugging.
- RFC 9421 HTTP Message Signature signer (`ed25519`, covered `@method`+`@path`+`@authority`+`content-digest`, label `sig1`, `created` window enforced by the coordinator). Symmetric to the platform's verifier; wire format verified by an inline oracle in the worker's own test suite.
- Keystore backends: libsecret (Secret Service) primary; encrypted-file (ChaCha20-Poly1305, key derived from `/etc/machine-id` + UID) fallback for headless hosts and containers.
- Local SQLite state at `$XDG_STATE_HOME/auspexai-worker/worker.db` with a sequential migration framework matching the coordinator's convention.
- `systemd --user` service unit template at `packaging/systemd/auspexai-worker.service` with Phase 1 hardening (`PrivateTmp`, `ProtectHome=read-only`, `ProtectSystem=strict`, `NoNewPrivileges`, `SystemCallFilter=@system-service`). ExecStart runs the heartbeat daemon.
- 73 tests on Python 3.11 + 3.12; ruff check + format-check clean.

Subsequent milestones (per design doc §14):

- **M3** — Assignment pull + manifest pinning + sensitive-content gate + `queue` / `peek` / `accept` / `refuse` / tenant allow-deny CLI.
- **M4** — Sandboxed runner subprocess (bubblewrap) + end-to-end work-unit execution.
- **M5** — Receipts store + `receipts list/show` + `log` CLI.
- **M6** — T1 upgrade via OAuth Device Flow + `withdraw` flow.
- **M7** — Cosign-signed `.deb` + source tarball as GitHub release artifacts.

Full design rationale: `Documentation/AuspexAI/v0.1.0/worker_daemon_design.md` (ratified into principles doc §5.19 on 2026-05-20).

## Scope

The Worker:

- Runs on volunteer machines — Linux x86_64 + ARM64 in Phase 1; macOS via Homebrew tap and Windows via WSL2 in Phase 2 (per §5.13)
- Generates an Ed25519 keypair on first run, stored in OS-native keystore (libsecret on Linux; Keychain / DPAPI in Phase 2 platforms) — volunteers never paste keys into web forms
- Enrolls anonymously (T0) or upgrades to verified identity via OAuth 2.0 Device Authorization Flow (T1+; M6)
- Executes work units within sandboxed limits (CPU, GPU, RAM, network caps; idle-only mode by default — Phase 2)
- Submits signed results (M4)
- Maintains a local audit log of what it ran, when, and for which manifest hash (M3+)

Worker trust tiers (T0 anonymous through T4 maintainer) govern work eligibility and quorum weight; see the AuspexAI Principles & Scope §6 for the trust model.

## Development

Requires Python 3.11+. Quick start:

```bash
uv venv
uv pip install -e ".[dev]"
auspexai-worker status                              # shows "not enrolled"
AUSPEXAI_COORDINATOR_URL=http://127.0.0.1:8080 \
  auspexai-worker bootstrap                         # enrolls against a local coordinator
AUSPEXAI_COORDINATOR_URL=http://127.0.0.1:8080 \
  auspexai-worker daemon --max-ticks=3              # run 3 heartbeats then exit
pytest                                              # 73 tests
ruff check src tests
ruff format --check src tests
```

## Conduct on the network

Worker conduct on the AuspexAI network is governed by the **Volunteer Terms of Participation** (forthcoming, Phase 2). Network-level abuse — running malicious workers, abusing reputation systems, attempting to influence quorum — is handled through technical enforcement (revocation, ban, trust-tier demotion), not Code-of-Conduct enforcement.

## License

[AGPL-3.0](LICENSE) — workers are network-served clients of the AuspexAI coordinator and inherit the platform's copyleft posture.

## Governance & policies

- [Governance](https://github.com/auspexai/.github/blob/main/GOVERNANCE.md) — roles, decision rules, recruitment, conflict of interest
- [Code of Conduct](https://github.com/auspexai/.github/blob/main/CODE_OF_CONDUCT.md) — community standards, reporting, escalation pathway
- [Contributing](https://github.com/auspexai/.github/blob/main/CONTRIBUTING.md) — DCO sign-off, PR workflow, RFC requirement for substantial architectural changes
- [Research Ethics Policy](https://github.com/auspexai/.github/blob/main/RESEARCH_ETHICS_POLICY.md) — what AI safety research can run on the network and how it's reviewed

## Watch this repo

Activity will begin as Phase 1 ramps up.
