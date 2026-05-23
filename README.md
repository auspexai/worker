# AuspexAI Worker

The volunteer worker process for [AuspexAI](https://github.com/auspexai) — runs on volunteer machines, executes work units dispatched by the coordinator, donates compute to AI safety research.

## Status

**Phase 1 — M1 + M2 + M3 + M4 SHIPPED 2026-05-20.** First-run enrollment, signed-heartbeat loop, assignment-pull pipeline, and sandboxed runner subprocess all verified end-to-end against the coordinator. Phase 1 worker target is Linux x86_64 + ARM64 (per principles doc §5.13 + §5.19); macOS/WSL2 packaging arrives in Phase 2.

What's live:

- `auspexai-worker bootstrap` — generates an Ed25519 keypair, persists to the OS keyring (libsecret) or an encrypted-file fallback, enrolls anonymously (T0) with the coordinator via `POST /api/v0/workers/enroll`, records the assigned `worker_id` in the local SQLite state DB. Idempotent on re-run.
- `auspexai-worker status` — shows worker_id, tier, pubkey fingerprint, enrollment timestamp, last heartbeat.
- `auspexai-worker daemon` — runs **two threads concurrently**: a heartbeat loop posting signed `POST /api/v0/workers/{id}/heartbeat` with capabilities (OS, arch, RAM total, CPU count, GPU observation + volunteer declaration, declared resource caps), and an assignment poller calling signed `GET /api/v0/workers/{id}/assignments` and running the M3 gate pipeline on every assignment received. SIGTERM/SIGINT trigger a clean stop on both. `--max-ticks=N` / `--verbose`.
- **Assignment-handling pipeline (M3):**
  - Manifest-pin defense (§5.14): first sighting of an experiment locks its `manifest_sha256`; a later assignment under a different hash is refused as a manifest-swap attempt.
  - Sensitive-content gate (§5.14 + §5.12): assignments carrying `sensitive_content_flags` in the payload require an explicit `accept <experiment-id>` to proceed (default-decline). *Note: the M6-era coordinator does NOT yet ship this field on `WorkUnitEnvelopeOut`; the gate is plumbed and waits for the coordinator-side change.*
  - Tenant allow/deny gate (§5.14): tenants on the deny list are always refused; a non-empty allow list restricts acceptance to listed tenants.
  - **Refusals call back to the coordinator** via `POST /api/v0/workers/{id}/assignments/{unit_id}/refuse` so the operator console can see the refusal reason and the unit's replication slot is freed for another worker (Option A per Q-W4). Each decision is also written to a local `assignment_audit` log.
- **M4: sandboxed runner + signed result submission.** On accept, the daemon spawns `auspexai-worker-runner` inside a `bubblewrap` sandbox (Phase 1 permissive policy: `--die-with-parent --new-session --dev-bind / /` plus env passthrough). The runner reads the work-unit envelope from stdin, executes the M4 *synthetic* executor (echoes the payload — tenant code arrives later), writes a Result body to `$AUSPEXAI_OUTPUT_PATH`, exits. Daemon reads the output, signs it with the worker key over a canonical encoding, POSTs to `/api/v0/workers/{id}/assignments/{unit_id}/result`. Local `submitted_results` table mirrors the coordinator's ack. Per Q-W4: any failure during dispatch (runner crash, submit error) triggers an explicit `refuse` to the coordinator. Per Q-W9: one-at-a-time on the poller thread; heartbeat thread runs independently. Per Q-W10: bubblewrap requires unprivileged user namespaces — on AppArmor-restricted hosts (Ubuntu 24.04 default) deploy needs `kernel.apparmor_restrict_unprivileged_userns=0` or a worker AppArmor profile. Passthrough mode (`[sandbox] use_bubblewrap = false`) is the documented escape hatch for CI + dev hosts.
- `auspexai-worker abort <unit-id>` — reads the runner PID file from the workspace, sends SIGTERM, waits `--grace-seconds` (default 5), sends SIGKILL if still running. No-op when no workspace / no PID / process already exited; always writes an audit row.
- M3 CLI verbs: `queue`, `peek <unit-id>`, `accept <coordinator-experiment-id>`, `refuse <unit-id>`, `tenant {allow,deny,list} <tenant-id>`.
- RFC 9421 HTTP Message Signature signer (`ed25519`, covered `@method`+`@path`+`@authority`+`content-digest`, label `sig1`, `created` window enforced by the coordinator). Symmetric to the platform's verifier; wire format verified by an inline oracle in the worker's own test suite.
- Keystore backends: libsecret (Secret Service) primary; encrypted-file (ChaCha20-Poly1305, key derived from `/etc/machine-id` + UID) fallback for headless hosts and containers.
- Local SQLite state at `$XDG_STATE_HOME/auspexai-worker/worker.db` with a sequential migration framework matching the coordinator's convention. Re-entrant transaction lock so the two daemon threads can write concurrently without colliding on `BEGIN`.
- `systemd --user` service unit templates with Phase 1 hardening (`PrivateTmp`, `ProtectHome=read-only`, `ProtectSystem=strict`, `NoNewPrivileges`, `SystemCallFilter=@system-service`). Two variants ship: `packaging/systemd/auspexai-worker.service` for pip/uv installs (ExecStart=`%h/.local/bin/auspexai-worker`); `packaging/systemd-deb/auspexai-worker.service` for `.deb` installs (ExecStart=`/opt/auspexai-worker/bin/auspexai-worker`, installed by the `.deb` to `/etc/systemd/user/auspexai-worker.service`).
- 157 tests on Python 3.11 + 3.12; ruff check + format-check clean.

Subsequent milestones (per design doc §14):
- **M5** ✅ Receipts store on `submitted_results` + `receipts list/show` + `log` CLI (SQLite-as-canonical-store per 2026-05-22 design update).
- **M6** ✅ T1 upgrade via OAuth Device Flow (`auspexai-worker login`) + `withdraw` flow (`auspexai-worker withdraw`).
- **M7** ✅ SHIPPED 2026-05-23 — Cosign-signed `.deb` + source tarball + wheel as GitHub release artifacts. `dh-virtualenv`-built `/opt/auspexai-worker/` venv; system-installed user-unit at `/etc/systemd/user/`; AppArmor profile confining the daemon with a `cx -> bwrap_sandbox` child profile that grants `userns,` scoped to bwrap-as-child-of-the-daemon (Q-W10 Phase 2 durable fix); postinst probes the sandbox as `nobody` before declaring install successful. Release pipeline at `.github/workflows/release.yml` signs via Sigstore keyless GitHub Actions OIDC (no manual cosign step per release). See `AUTHORIZED_SIGNERS.md` for the trust roster.

Full design rationale: `Documentation/AuspexAI/v0.1.0/worker_daemon_design.md` (ratified into principles doc §5.19 on 2026-05-20).

## Install

Phase 1 supported target: **Ubuntu 24.04+ / Debian 12+, x86_64**. ARM64 + Ubuntu 22.04 land in Phase 2.

### Via `.deb` release artifact (recommended)

Download the `.deb` and its Cosign signature/cert from the [latest release](https://github.com/auspexai/worker/releases/latest). **Verify before installing** (the signature anchors at the Maintainer's GitHub OIDC identity via Sigstore — no long-lived signing key to compromise):

```bash
# One-time: install cosign if you don't already have it
# (https://docs.sigstore.dev/cosign/installation)

# Verify the .deb
cosign verify-blob \
  --certificate-identity-regexp='^https://github\.com/auspexai/.+/\.github/workflows/.+@.+$' \
  --certificate-oidc-issuer='https://token.actions.githubusercontent.com' \
  --signature auspexai-worker_<version>_amd64.deb.sig \
  --certificate auspexai-worker_<version>_amd64.deb.cert \
  auspexai-worker_<version>_amd64.deb

# Install
sudo apt install ./auspexai-worker_<version>_amd64.deb
```

The postinst will reload the AppArmor profile and probe the sandbox as user `nobody`. If the probe fails, the install fails with an actionable error pointing at either the sysctl bridge (`kernel.apparmor_restrict_unprivileged_userns=0` for lab use) or the AppArmor profile path (durable fix on Ubuntu 24.04+).

After install, **each volunteer user runs (as themselves, not root):**

```bash
systemctl --user enable --now auspexai-worker.service
auspexai-worker status       # confirms enrollment (T0 anonymous)
auspexai-worker login        # binds to GitHub identity (T1)
```

Tail the daemon: `journalctl --user -u auspexai-worker -f`.

### Trust roster

The list of identities authorized to sign release artifacts and contribution receipts lives at [`auspexai/.github/security/AUTHORIZED_SIGNERS.md`](https://github.com/auspexai/.github/blob/main/security/AUTHORIZED_SIGNERS.md). Verifiers should match the cosign `--certificate-identity-regexp` against the entries in that file.

### Build the `.deb` from source

```bash
packaging/build-deb.sh            # builds in a podman/docker Ubuntu 24.04 container
packaging/build-deb.sh --test     # also install-tests in a second clean container
```

Outputs land at `/tmp/auspexai-deb-build/auspexai-worker_<version>_amd64.deb`.

## Scope

The Worker:

- Runs on volunteer machines — Linux x86_64 + ARM64 in Phase 1; macOS via Homebrew tap and Windows via WSL2 in Phase 2 (per §5.13)
- Generates an Ed25519 keypair on first run, stored in OS-native keystore (libsecret on Linux; Keychain / DPAPI in Phase 2 platforms) — volunteers never paste keys into web forms
- Enrolls anonymously (T0) or upgrades to verified identity via OAuth 2.0 Device Authorization Flow (T1+; M6)
- Executes work units within sandboxed limits (CPU, GPU, RAM, network caps; idle-only mode by default — Phase 2)
- Submits signed results (M4)
- Maintains a local audit log of what it ran, when, and for which manifest hash (M3+)

Worker trust tiers (T0 anonymous through T4 maintainer) govern work eligibility and quorum weight; see the AuspexAI Principles & Scope §6 for the trust model.

## Identity binding (T0 → T1)

A fresh install enrolls as **T0 anonymous** with no user interaction. To bind the worker to a verifiable identity and unlock higher-trust roles (vouching, Approver eligibility, unique work-unit assignments) run:

```
auspexai-worker login
```

This launches an [OAuth 2.0 Device Authorization Flow](https://datatracker.ietf.org/doc/html/rfc8628) against GitHub:

1. The worker prints a short code and a verification URL (`github.com/login/device`).
2. Open the URL in any browser (any device — phone, laptop, the same machine — your choice), sign in to GitHub, enter the code.
3. GitHub returns an access token to the worker; the worker hands it to the AuspexAI coordinator, which verifies it with GitHub and mints a one-shot binding token; the worker then exchanges that binding token for a T1 upgrade. The access token is **never stored** on the worker — it's discarded as soon as the binding completes.

The `read:user` scope is the only scope ever requested. Subsequent worker↔coordinator authentication continues to use the worker's own Ed25519 keypair, not the GitHub token.

### Email-fallback for institutional contexts

Phase 1 is **GitHub-only** for the OAuth IdP. Institutional volunteers, researchers, or workers running in environments where GitHub Device Flow isn't usable can request an alternative binding path by emailing **contact@auspexai.network**. A formal email-based binding path is on the Phase 2 roadmap (per principles doc §5.10 + §9 Q5); pre-Phase-2 accommodations are case-by-case under the Maintainer's discretion.

## Withdrawal

To retire a worker and remove its local state:

```
auspexai-worker withdraw
```

The command prints what will be deleted, prompts for explicit confirmation (you must type the word `withdraw`), then:

1. Calls the coordinator's retire endpoint so the scheduler stops handing it work.
2. Deletes the local SQLite state DB (`worker.db`) — audit log, manifest pins, receipts, tenant lists, consent rows.
3. Deletes the worker's Ed25519 keypair from the keystore.

**Receipts already issued by the coordinator REMAIN in the coordinator's transparency log per Principles & Scope §5.15.** They stay signed and verifiable but become unattributed (severable-PII pattern). The CLI prints this verbatim at the confirmation prompt so the volunteer is reminded of the consent terms at the moment of withdrawal.

After withdrawal completes, follow up with your package manager (`apt remove auspexai-worker`, `pipx uninstall auspexai-worker`, etc.) to finish the uninstall.

## Development

Requires Python 3.11+. Quick start:

```bash
uv venv
uv pip install -e ".[dev]"
auspexai-worker status                              # shows "not enrolled"
auspexai-worker bootstrap                           # enrolls against the
                                                    # public coord (default
                                                    # since v0.1.2)
auspexai-worker daemon --max-ticks=3                # run 3 heartbeats then exit
pytest
ruff check src tests
ruff format --check src tests
```

Running against a coordinator on your own machine (lab mode) — override
the default `https://coord.auspexai.network`:

```bash
AUSPEXAI_COORDINATOR_URL=http://127.0.0.1:8080 \
  auspexai-worker bootstrap
AUSPEXAI_COORDINATOR_URL=http://127.0.0.1:8080 \
  auspexai-worker daemon --max-ticks=3
```

Or persist via TOML at `~/.config/auspexai-worker/worker.toml`:

```toml
[coordinator]
url = "http://127.0.0.1:8080"
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
