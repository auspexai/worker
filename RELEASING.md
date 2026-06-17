# Releasing the AuspexAI worker

How a worker release is cut. **Ownership:** Claude cuts the release (commit → tag → push); the **USER rolls** the fleet (to dogfood the install/update UX). See the deploy-ownership boundary in project memory.

Versioning is **hatch-vcs** — the version is derived from the git tag, so the tag must sit on a clean `HEAD`.

---

## 1. Pre-flight gate — ALL must pass before tagging

1. **On `main`**, working tree contains only the intended changes.
2. **`make ci` green** — `sync` + `ruff check` + `ruff format --check` + the full test suite + `uv build`. (This mirrors `ci.yml` step-for-step; a green `make ci` ≈ a green CI run.)
3. **`make redteam` → `7 passed`** — the full §41(a) STRICT escape proof, all 5 vectors under real cgroup delegation. **This is a hard gate: do not ship a worker release if the sandbox red-team is red.**
   - **Equivalent check:** confirm the latest `redteam.yml` run on `main` is green — `gh run list --workflow=redteam.yml -L1`. The self-hosted `virt-mayhem` runner re-proves 7/7 on every push + nightly, so a recent green run on the release commit satisfies this.
   - `make redteam` needs cgroup delegation; on a box without it (a plain dev host) it **fails loud** (no silent skip) rather than passing vacuously — run it on rage or virt-mayhem, or rely on the `redteam.yml` run.

## 2. Cut (Claude)

4. Commit everything (tree must be clean — hatch-vcs reads the version from the tag on `HEAD`; the release workflow's guard rejects a dev/dirty version).
5. `git tag -a vX.Y.Z -m "worker vX.Y.Z — <summary>"` then `git push origin main && git push origin vX.Y.Z`.
6. The tag fires `release.yml`: build → cosign-sign → **draft** GitHub release with assets → flip to **published** (this fires the coordinator webhook → fleet update banner) → PyPI trusted publish.

## 3. Post-cut verification

7. **Confirm `ci.yml` is green** on the `main` push. `release.yml` does **not** run tests, so `ci.yml` is the test gate — a red CI on a freshly-published release means fix-forward (precedent: v0.2.18 shipped red → v0.2.19 fixed `os.memfd_create` portability + greened CI).
8. Confirm the **GitHub release published** and **PyPI** shows the new version (`curl -s https://pypi.org/pypi/auspexai-worker/json | jq -r .info.version`).

## 4. Roll (USER)

9. USER rolls the fleet via the installer. **STRICT workers (e.g. mayhem0) need a full installer re-run** (`curl … | bash`), **not** a pip/binary bump — only the installer re-lays the systemd unit (`Delegate=yes`), the AppArmor cgroup rule, and `libseccomp2`, which is what **activates the cgroup resource caps**.
10. Verify the caps activated on a STRICT box:
    ```
    systemctl --user show auspexai-worker -p Delegate -p DelegateControllers
    ```
    Expect `Delegate=yes` with `memory` + `pids` in `DelegateControllers`. (They degrade silently to the rlimit floor otherwise.)

---

_This checklist exists so the red-team gate (step 3) doesn't depend on remembering it. Until/while `virt-mayhem` is healthy it's automatic on every push; this is the belt-and-suspenders for the manual path._
