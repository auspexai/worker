#!/usr/bin/env bash
#
# AuspexAI worker installer
#
# Usage:
#   curl -sSL https://getworker.auspexai.network | bash
#   curl -sSL https://getworker.auspexai.network | bash -s -- --version 0.1.5
#
# What this does:
#   1. Checks prerequisites (Python 3.11+, bubblewrap)
#   2. Downloads the latest release from GitHub
#   3. Fast path: installs .deb if one exists for this arch
#   4. Fallback: creates a venv at /opt/auspexai-worker, pip-installs
#      the wheel, and lays down the systemd unit + AppArmor profile
#   5. Runs the sandbox probe to verify everything works
#
# Requires: bash, curl, Python >= 3.11, sudo
# Optional: cosign (for signature verification), bubblewrap (for sandbox)

set -euo pipefail

TMPDIR_CLEANUP=""
trap 'rm -rf "$TMPDIR_CLEANUP"' EXIT

INSTALL_PREFIX="/opt/auspexai-worker"
SYSTEMD_UNIT_DIR="/etc/systemd/user"
APPARMOR_DIR="/etc/apparmor.d"
GITHUB_REPO="auspexai/worker"
MIN_PYTHON_MINOR=11

# ONE installer, OS-aware internals (NOT a separate script per OS): Linux uses
# apt/.deb + systemd; macOS uses pip + a launchd LaunchAgent. Everything branches
# on this single value.
OS="$(uname -s)"  # Linux | Darwin (macOS)

# ── Helpers ──────────────────────────────────────────────────────────

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m==> WARNING:\033[0m %s\n' "$*" >&2; }
fail()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "'$1' is required but not found. Install it and retry."
}

# §41(a) STRICT sandbox runtime dependencies: bubblewrap (constructs the
# sandbox) + libseccomp (loaded by pyseccomp via ctypes for the syscall-denylist
# gate). Under strict, the daemon FAILS CLOSED on a unit if either is missing —
# so install them here. apt-based hosts get an active install; elsewhere we warn
# with the package names. Non-fatal: a permissive worker doesn't need either.
ensure_sandbox_deps() {
    local policy="$1"
    # macOS has no bubblewrap/libseccomp; STRICT isolation there is a separate
    # mechanism (sandbox-exec — coming). Nothing apt-installable here; the worker
    # runs permissive on macOS until that lands.
    [ "$OS" = "Darwin" ] && return 0
    local missing=()
    command -v bwrap >/dev/null 2>&1 || missing+=(bubblewrap)
    # libseccomp.so.2 ships in libseccomp2 (Debian/Ubuntu); detect via ldconfig.
    if ! ldconfig -p 2>/dev/null | grep -q 'libseccomp\.so'; then
        missing+=(libseccomp2)
    fi
    if [ ${#missing[@]} -eq 0 ]; then
        return 0
    fi
    if [ "$policy" = "strict" ]; then
        if command -v apt >/dev/null 2>&1; then
            info "Installing STRICT sandbox deps: ${missing[*]} …"
            sudo apt install -y "${missing[@]}" \
                || warn "could not install ${missing[*]} — STRICT units will be refused until present (install manually)"
        else
            warn "STRICT needs: ${missing[*]} (no apt found — install them with your package manager, or the worker will refuse STRICT units)"
        fi
    else
        warn "sandbox deps not installed: ${missing[*]} — needed if you switch [sandbox] policy = strict (install: sudo apt install ${missing[*]})"
    fi
}

# ── Flavors (§9 #46) ─────────────────────────────────────────────────
# Install profiles, defined HERE as data (this script is served live from
# main via getworker.auspexai.network, so the registry can't be a sibling
# file). One codebase, one release train — a flavor only changes what the
# onramp installs/configures. Adding a flavor = new case arms, no engine
# changes. The worker records its flavor in worker.toml so upgrades
# preserve it; scheduling stays capability-based (flavor is NOT a routing
# key).

FLAVOR_NAMES="lean inference full"

flavor_desc() {
    case "$1" in
        lean)      echo "minimal worker — synthetic + staged work only (default)" ;;
        inference) echo "serves local models to experiments — installs Ollama (a local model server) + model tooling, and enables serving on this machine" ;;
        full)      echo "everything: inference serving + all optional extras" ;;
    esac
}

# pip packages installed into the worker venv for this flavor
flavor_pip() {
    case "$1" in
        inference|full) echo "huggingface_hub>=0.20" ;;
    esac
}

# system-level installs for this flavor (each token has an install_<token> fn)
flavor_system() {
    case "$1" in
        inference|full) echo "ollama" ;;
    esac
}

# worker.toml settings applied for this flavor (token = setter shorthand).
# EVERY flavor states its serving posture explicitly so switching DOWN
# (e.g. inference → lean) actively disables serving rather than silently
# inheriting the prior choice — flavors are declarative, not additive,
# for config. (System installs stay additive: we never uninstall Ollama;
# the volunteer's machine may use it for other things.)
flavor_config() {
    case "$1" in
        lean)           echo "inference-backend=none" ;;
        inference|full) echo "inference-backend=ollama" ;;
    esac
}

flavor_valid() {
    case " ${FLAVOR_NAMES} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

list_flavors() {
    echo "Available flavors:"
    for f in $FLAVOR_NAMES; do
        printf '  %-10s %s\n' "$f" "$(flavor_desc "$f")"
    done
}

# Idempotent Ollama install via the official installer (arm64/Jetson-capable;
# sets up its own systemd service). Failure is NON-FATAL: the worker still
# installs, the dashboard's reachability badge shows the gap, and re-running
# this installer heals it. The flavor choice IS the volunteer's consent to
# this third-party install — the menu text says so plainly.
install_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        info "Ollama already installed ($(ollama --version 2>/dev/null || echo 'version unknown')) — skipping install"
        # Installed ≠ serving: a pre-existing Ollama can have a broken/stale
        # service (seen live: a leftover systemd override with a bad
        # OLLAMA_MODELS crash-looped `ollama serve` while the binary looked
        # fine). Verify it actually answers before declaring this flavor good.
        if curl -fsS -m 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
            info "Ollama serving: $(curl -fsS -m 3 http://127.0.0.1:11434/api/version 2>/dev/null)"
        else
            flavor_issue "Ollama is installed but NOT serving on 127.0.0.1:11434 — inference stays unavailable until it runs. Check: systemctl status ollama (and any /etc/systemd/system/ollama.service.d/ overrides)"
        fi
        return 0
    fi
    info "Installing Ollama …"
    if [ "$OS" = "Darwin" ]; then
        # ollama.com/install.sh is the LINUX installer (it half-installs the CLI then
        # tries to launch a desktop app that isn't there). On macOS use Homebrew —
        # the headless `ollama` formula + a brew service that serves on :11434.
        if ! command -v brew >/dev/null 2>&1; then
            flavor_issue "Ollama isn't installed and Homebrew wasn't found. Install Ollama from https://ollama.com/download (or 'brew install ollama'), then re-run this installer (inference flavor) to finish."
            return 0
        fi
        if ! brew install ollama; then
            flavor_issue "'brew install ollama' failed — install Ollama from https://ollama.com/download, then re-run this installer (inference flavor)."
            return 0
        fi
        if ! brew services start ollama >/dev/null 2>&1; then
            nohup ollama serve >/dev/null 2>&1 &
        fi
    else
        info "(official installer from ollama.com; can be a large download)"
        if ! curl -fsSL https://ollama.com/install.sh | sh; then
            flavor_issue "Ollama install failed — inference serving stays unavailable until Ollama is present. Re-run this installer (same flavor) to retry."
            return 0
        fi
    fi
    # Surface the installed version (determinism provenance; the daemon
    # re-reports it in heartbeat capabilities once serving).
    local tries=0
    while [ $tries -lt 15 ]; do
        if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
            info "Ollama serving: $(curl -fsS http://127.0.0.1:11434/api/version 2>/dev/null)"
            return 0
        fi
        tries=$((tries + 1))
        sleep 2
    done
    flavor_issue "Ollama installed but not reachable on 127.0.0.1:11434 yet — it may still be starting. Verify later: curl 127.0.0.1:11434/api/version"
}

# Issues hit while applying the flavor (pip/system/config steps). Each is
# warned inline when it happens, but inline warns scroll away in a long
# install — the footer re-prints them as one loud summary so the volunteer
# can't end up with a silently half-applied flavor (seen live: a broken
# pre-existing Ollama service hid behind a clean-looking install).
FLAVOR_ISSUES=""

flavor_issue() {
    warn "$1"
    FLAVOR_ISSUES="${FLAVOR_ISSUES}  - $1\n"
}

# Apply the chosen flavor AFTER the package install, BEFORE bootstrap/start
# ([inference] is read at daemon start). Works identically on the deb and
# wheel paths — both land the venv at ${INSTALL_PREFIX}.
apply_flavor() {
    local flavor="$1"
    info "Applying flavor: ${flavor} — $(flavor_desc "$flavor")"

    local pip_pkgs
    pip_pkgs=$(flavor_pip "$flavor")
    if [ -n "$pip_pkgs" ]; then
        if [ -x "${INSTALL_PREFIX}/bin/pip" ]; then
            info "Installing flavor pip packages: ${pip_pkgs} …"
            # shellcheck disable=SC2086 — word-splitting the package list is intended
            sudo "${INSTALL_PREFIX}/bin/pip" install -q $pip_pkgs \
                || flavor_issue "could not install pip packages (${pip_pkgs}); some flavor features may be unavailable"
        else
            flavor_issue "pip not found in ${INSTALL_PREFIX}; install ${pip_pkgs} manually"
        fi
    fi

    local sys_tokens
    sys_tokens=$(flavor_system "$flavor")
    for token in $sys_tokens; do
        "install_${token}"
    done

    # Config writes run AS THE USER (XDG config), via the worker CLI so the
    # volunteer's worker.toml is edited surgically. Guarded for --version
    # installs of pre-flavor binaries (< v0.2.0).
    if "${INSTALL_PREFIX}/bin/auspexai-worker" flavor --help >/dev/null 2>&1; then
        local cfg_tokens
        cfg_tokens=$(flavor_config "$flavor")
        for token in $cfg_tokens; do
            case "$token" in
                inference-backend=*)
                    local backend="${token#inference-backend=}"
                    "${INSTALL_PREFIX}/bin/auspexai-worker" inference set-backend "$backend" >/dev/null \
                        || flavor_issue "could not set [inference] backend = ${backend} in worker.toml"
                    if [ "$backend" = "none" ]; then
                        info "Inference serving disabled ([inference] backend = none)"
                    else
                        info "Enabled [inference] backend = ${backend}"
                    fi
                    ;;
            esac
        done
        # Always record the flavor (lean included) so upgrades preserve it.
        "${INSTALL_PREFIX}/bin/auspexai-worker" flavor set "$flavor" >/dev/null \
            || flavor_issue "could not record the flavor in worker.toml (upgrades won't remember it)"
    else
        warn "installed worker predates flavor support (< v0.2.0); flavor not recorded"
    fi

    # Down-switch note: system installs are additive (we never uninstall a
    # third-party tool the volunteer's machine may use) — say so when this
    # flavor doesn't use an Ollama that's still present.
    case " $(flavor_system "$flavor") " in
        *" ollama "*) ;;
        *)
            if command -v ollama >/dev/null 2>&1; then
                info "Note: Ollama remains installed (this flavor just doesn't use it)."
                info "      Remove it manually if unwanted: https://github.com/ollama/ollama/blob/main/docs/linux.md#uninstall"
            fi
            ;;
    esac
}

# Resolution: explicit --flavor > interactive menu (the volunteer's PRIOR
# choice, recorded in worker.toml, is the default — Enter keeps it; an
# update is also the natural moment to switch) > recorded silently when
# there's no tty > lean. The recorded-flavor grep is version-independent —
# works even when the OLD binary predates \`flavor show\`.
resolve_flavor() {
    local requested="$1"
    if [ -n "$requested" ]; then
        echo "$requested"
        return
    fi
    local recorded=""
    local toml="$HOME/.config/auspexai-worker/worker.toml"
    if [ -f "$toml" ]; then
        recorded=$(grep -E '^[[:space:]]*flavor[[:space:]]*=' "$toml" 2>/dev/null \
            | head -1 | sed 's/.*=[[:space:]]*"\{0,1\}//;s/"\{0,1\}[[:space:]]*$//') || true
        if [ -n "$recorded" ] && ! flavor_valid "$recorded"; then
            recorded=""
        fi
    fi
    # [ -r /dev/tty ] passes on permission bits even with NO controlling
    # terminal (ENXIO at open) — actually try opening it, or the read below
    # would abort the whole install under set -e.
    if ! (exec </dev/tty) 2>/dev/null; then
        # Non-interactive (no tty): keep the prior choice; first install
        # defaults to lean.
        if [ -n "$recorded" ]; then
            info "Keeping flavor: ${recorded} (recorded in worker.toml; pass --flavor to change)" >&2
            echo "$recorded"
        else
            echo "lean"
        fi
        return
    fi
    # Interactive: always offer the menu. Default = the prior choice when
    # one is recorded (Enter keeps it), else lean.
    local default_flavor="${recorded:-lean}"
    {
        echo ""
        if [ -n "$recorded" ]; then
            echo "Install flavor (Enter keeps your current choice):"
        else
            echo "Choose an install flavor:"
        fi
        local i=1
        for f in $FLAVOR_NAMES; do
            if [ "$f" = "$default_flavor" ] && [ -n "$recorded" ]; then
                printf '  %d) %-10s %s  (current)\n' "$i" "$f" "$(flavor_desc "$f")"
            else
                printf '  %d) %-10s %s\n' "$i" "$f" "$(flavor_desc "$f")"
            fi
            i=$((i + 1))
        done
        printf 'Flavor [%s]: ' "$default_flavor"
    } >&2
    local reply
    read -r reply </dev/tty || reply=""
    if [ -z "$reply" ]; then
        echo "$default_flavor"
        return
    fi
    local n=1
    for f in $FLAVOR_NAMES; do
        if [ "$reply" = "$n" ] || [ "$reply" = "$f" ]; then
            echo "$f"
            return
        fi
        n=$((n + 1))
    done
    echo "$default_flavor"
}

# §41: the volunteer's host-isolation choice for running tenant code. Mirrors
# resolve_sandbox_policy — interactive menu (prior choice is the default; Enter
# keeps it) > recorded silently when there's no tty > permissive. The volunteer is
# ASKED rather than silently defaulted, because this is the consent moment for
# running other people's code on their machine. Default permissive (STRICT-by-
# default was tried + reverted 2026-06-27 — it fails closed on hosts with restricted
# unprivileged user namespaces, which would strand volunteers); the menu still
# offers + describes strict ("narrow filesystem, no network, namespace-isolated").
resolve_sandbox_policy() {
    local recorded=""
    local toml="$HOME/.config/auspexai-worker/worker.toml"
    if [ -f "$toml" ]; then
        recorded=$(grep -E '^[[:space:]]*policy[[:space:]]*=' "$toml" 2>/dev/null \
            | head -1 | sed 's/.*=[[:space:]]*"\{0,1\}//;s/"\{0,1\}[[:space:]]*$//') || true
        case "$recorded" in permissive | strict) ;; *) recorded="" ;; esac
    fi
    local default_policy="${recorded:-permissive}"
    if ! (exec </dev/tty) 2>/dev/null; then
        [ -n "$recorded" ] && info "Keeping sandbox policy: ${recorded}" >&2
        echo "$default_policy"
        return
    fi
    {
        echo ""
        echo "This worker runs experiment code from researchers. How should it be isolated?"
        local strict_tag="" perm_tag=""
        [ "$default_policy" = "strict" ] && strict_tag="  (current)"
        [ "$default_policy" = "permissive" ] && [ -n "$recorded" ] && perm_tag="  (current)"
        echo "  1) strict      narrow filesystem, no network, namespace-isolated${strict_tag}"
        echo "  2) permissive  shares your host filesystem (only for fully-trusted setups)${perm_tag}"
        printf 'Sandbox policy [%s]: ' "$default_policy"
    } >&2
    local reply
    read -r reply </dev/tty || reply=""
    case "$reply" in
        "") echo "$default_policy" ;;
        1 | strict) echo "strict" ;;
        2 | permissive) echo "permissive" ;;
        *) echo "$default_policy" ;;
    esac
}

# M3 on-demand model acquisition: the volunteer's choice to let this worker
# DOWNLOAD the exact model an experiment needs when it doesn't already have it.
# Only meaningful for inference flavors (a lean worker serves no models) — returns
# "" for others so the caller skips it. Mirrors resolve_sandbox_policy: prior
# choice is the default (Enter keeps it); recorded silently with no tty. Default
# OFF — pulling models the network requests spends the volunteer's bandwidth +
# disk, so it's an explicit opt-in (same posture as the model-setup prompt).
resolve_auto_acquire() {
    local flavor="$1"
    case "$flavor" in inference | full) ;; *) echo ""; return ;; esac
    local recorded=""
    local toml="$HOME/.config/auspexai-worker/worker.toml"
    if [ -f "$toml" ]; then
        recorded=$(grep -E '^[[:space:]]*auto_acquire[[:space:]]*=' "$toml" 2>/dev/null \
            | head -1 | sed 's/.*=[[:space:]]*//;s/[[:space:]]*$//') || true
        case "$recorded" in true | false) ;; *) recorded="" ;; esac
    fi
    local default_aa="${recorded:-false}"
    if ! (exec </dev/tty) 2>/dev/null; then
        [ -n "$recorded" ] && info "Keeping auto-acquire: ${recorded}" >&2
        echo "$default_aa"
        return
    fi
    {
        echo ""
        echo "Allow this worker to auto-acquire models? When an experiment needs a model"
        echo "this worker doesn't have, the worker downloads that exact model on demand"
        echo "(your bandwidth + disk). Off = serve only models you set up here."
        local y_tag="" n_tag=""
        [ "$default_aa" = "true" ] && y_tag="  (current)"
        [ "$default_aa" = "false" ] && [ -n "$recorded" ] && n_tag="  (current)"
        echo "  1) no   only serve models already set up on this machine${n_tag}"
        echo "  2) yes  download requested models on demand${y_tag}"
        printf 'Auto-acquire models [%s]: ' "$([ "$default_aa" = "true" ] && echo yes || echo no)"
    } >&2
    local reply
    read -r reply </dev/tty || reply=""
    case "$reply" in
        "") echo "$default_aa" ;;
        1 | n | no | N | NO | No) echo "false" ;;
        2 | y | yes | Y | YES | Yes) echo "true" ;;
        *) echo "$default_aa" ;;
    esac
}

# ── Detect Python ────────────────────────────────────────────────────

find_python() {
    # Newest-versioned first, then the generic `python3` (most installs symlink it to
    # the latest — so a 3.13/3.14 install is found even when no pythonX.Y command exists
    # for it), then any other python3.NN on PATH as a final catch-all. The first one
    # that is >= 3.${MIN_PYTHON_MINOR} wins; an older one (e.g. macOS's system 3.9) is
    # skipped rather than failing the install.
    local candidate ver
    local candidates="python3.14 python3.13 python3.12 python3.11 python3"
    # Append any remaining python3.NN executables on PATH (deduped), so we don't depend
    # on this list being exhaustive as new minors ship.
    local extra
    extra=$(ls -1 $(echo "$PATH" | tr ':' ' ') 2>/dev/null \
        | grep -E '^python3\.[0-9]+$' | sort -u || true)
    [ -n "$extra" ] && candidates="$candidates $extra"
    for candidate in $candidates; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        ver=$("$candidate" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null) || continue
        if [ "$ver" -ge "$MIN_PYTHON_MINOR" ] 2>/dev/null; then
            echo "$candidate"
            return
        fi
    done
    return 1
}

# ── Detect architecture (matches Debian naming) ─────────────────────

detect_arch() {
    local machine
    machine=$(uname -m)
    case "$machine" in
        x86_64)  echo "amd64" ;;
        aarch64) echo "arm64" ;;
        armv7l)  echo "armhf" ;;
        *)       echo "$machine" ;;
    esac
}

# ── Fetch latest release info from GitHub ────────────────────────────

fetch_release() {
    local version="$1"
    local api_url

    if [ -n "$version" ]; then
        api_url="https://api.github.com/repos/${GITHUB_REPO}/releases/tags/v${version}"
    else
        api_url="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"
    fi

    curl -fsSL "$api_url" 2>/dev/null || fail "could not fetch release info from GitHub"
}

# ── Main ─────────────────────────────────────────────────────────────

do_uninstall() {
    info "Uninstalling AuspexAI worker …"

    # Stop and disable systemd unit
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --user stop auspexai-worker.service 2>/dev/null || true
        systemctl --user disable auspexai-worker.service 2>/dev/null || true
    fi

    # Withdraw from coordinator if enrolled
    if [ -x "${INSTALL_PREFIX}/bin/auspexai-worker" ]; then
        local enrolled
        enrolled=$("${INSTALL_PREFIX}/bin/auspexai-worker" status 2>&1 | grep -c "worker-id:" || true)
        if [ "$enrolled" != "0" ]; then
            printf 'De-enroll from coordinator before removing? [Y/n] '
            read -r reply </dev/tty
            case "$reply" in
                n|N|no|NO) ;;
                *) "${INSTALL_PREFIX}/bin/auspexai-worker" withdraw --yes 2>/dev/null || warn "withdraw failed; continuing with local removal" ;;
            esac
        fi
    fi

    # Check if installed via deb
    if dpkg -s auspexai-worker >/dev/null 2>&1; then
        info "Removing .deb package …"
        sudo apt remove -y auspexai-worker
    else
        # Remove pip-installed artifacts
        info "Removing ${INSTALL_PREFIX} …"
        sudo rm -rf "${INSTALL_PREFIX}"
        sudo rm -f /usr/local/bin/auspexai-worker
        sudo rm -f "${SYSTEMD_UNIT_DIR}/auspexai-worker.service"
        sudo rm -f "${APPARMOR_DIR}/auspexai-worker"
        if [ -x /sbin/apparmor_parser ] && [ -f /sys/module/apparmor/parameters/enabled ] && \
           [ "$(cat /sys/module/apparmor/parameters/enabled 2>/dev/null)" = "Y" ]; then
            sudo apparmor_parser -R "${APPARMOR_DIR}/auspexai-worker" 2>/dev/null || true
        fi
    fi

    # Reload systemd
    if command -v systemctl >/dev/null 2>&1; then
        sudo systemctl daemon-reload 2>/dev/null || true
    fi

    info "Uninstalled."
    echo ""
    if command -v ollama >/dev/null 2>&1; then
        echo "Ollama (installed by the inference flavor, or already present) was NOT"
        echo "removed — your machine may use it for other things. To remove it, see"
        echo "https://github.com/ollama/ollama/blob/main/docs/linux.md#uninstall"
    fi
    echo "Local state at ~/.local/state/auspexai-worker/ was NOT removed."
    models_dir="$HOME/.local/share/auspexai-worker/models"
    if [ -d "$models_dir" ] && [ -n "$(ls -A "$models_dir" 2>/dev/null)" ]; then
        echo "Downloaded models retained under $models_dir (preserved across installs/upgrades; can be many GB):"
        ls -1 "$models_dir" 2>/dev/null | sed 's/^/    - /'
    else
        echo "Downloaded models (if any) live under ~/.local/share/auspexai-worker/models/."
    fi
    echo "To remove everything: rm -rf ~/.local/state/auspexai-worker ~/.local/share/auspexai-worker"
}

# Start (and enable-at-boot) the worker service — launchd on macOS, systemd on
# Linux; both fall back to a detached daemon if the service manager is unavailable.
start_worker_service() {
    if [ "$OS" = "Darwin" ]; then
        local plist="$HOME/Library/LaunchAgents/network.auspexai.worker.plist"
        launchctl unload "$plist" 2>/dev/null || true
        if launchctl load "$plist" 2>/dev/null; then
            info "Worker loaded via launchd (auto-starts on login)."
        else
            info "launchctl load failed; starting the daemon directly …"
            nohup "${INSTALL_PREFIX}/bin/auspexai-worker" daemon </dev/null >/dev/null 2>&1 &
            info "daemon started (pid $!); logs: auspexai-worker logs -f"
        fi
    elif ! systemctl --user enable --now auspexai-worker.service 2>/dev/null; then
        info "systemd user service unavailable; starting the daemon directly …"
        nohup "${INSTALL_PREFIX}/bin/auspexai-worker" daemon </dev/null >/dev/null 2>&1 &
        info "daemon started (pid $!); logs at: auspexai-worker logs -f"
    fi
}

main() {
    local requested_version=""
    local requested_flavor=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --version) requested_version="$2"; shift 2 ;;
            --flavor)
                requested_flavor="$2"
                flavor_valid "$requested_flavor" \
                    || fail "unknown flavor: ${requested_flavor} (one of: ${FLAVOR_NAMES})"
                shift 2
                ;;
            --list-flavors)
                list_flavors
                exit 0
                ;;
            --uninstall)
                do_uninstall
                exit 0
                ;;
            --help|-h)
                echo "Usage: install.sh [--version VERSION] [--flavor NAME] [--uninstall]"
                echo ""
                echo "Installs the AuspexAI worker from the latest GitHub release."
                echo "If a .deb exists for this architecture, it is preferred."
                echo "Otherwise, a pip-based install into /opt/auspexai-worker/ is used."
                echo ""
                echo "  --uninstall      Stop, de-enroll, and remove the worker"
                echo "  --version V      Install a specific version instead of latest"
                echo "  --flavor NAME    Install profile (upgrades keep the recorded one)"
                echo "  --list-flavors   Show available flavors and exit"
                echo ""
                list_flavors
                exit 0
                ;;
            *) fail "unknown option: $1" ;;
        esac
    done

    need_cmd curl
    need_cmd sudo

    # ── Check for existing install ───────────────────────────────────

    if [ -x "${INSTALL_PREFIX}/bin/auspexai-worker" ]; then
        local current
        current=$("${INSTALL_PREFIX}/bin/auspexai-worker" --version 2>/dev/null || echo "unknown")
        warn "existing install detected at ${INSTALL_PREFIX} (${current})"
        printf '    Continue and upgrade? [y/N] '
        read -r reply </dev/tty
        case "$reply" in
            y|Y|yes|YES) ;;
            *) echo "Aborted."; exit 0 ;;
        esac

        # Stop the running daemon before overwriting binaries
        if systemctl --user is-active auspexai-worker.service >/dev/null 2>&1; then
            info "Stopping systemd service …"
            systemctl --user stop auspexai-worker.service
        elif pgrep -f 'auspexai-worker daemon' >/dev/null 2>&1; then
            info "Stopping running daemon (pid $(pgrep -f 'auspexai-worker daemon')) …"
            pkill -f 'auspexai-worker daemon' 2>/dev/null || true
            sleep 1
        fi
    fi

    # ── Resolve the install flavor (§9 #46) ──────────────────────────
    # One up-front decision: explicit flag > recorded (upgrades preserve the
    # volunteer's choice) > interactive menu > lean. The inference flavor's
    # menu text states plainly that it installs Ollama — choosing it IS the
    # consent for that third-party install.

    local flavor
    flavor=$(resolve_flavor "$requested_flavor")
    info "Flavor: ${flavor}"

    # §41: ask the volunteer how to isolate tenant code (consent moment).
    local sandbox_policy
    sandbox_policy=$(resolve_sandbox_policy)
    # macOS strict isn't available yet (no bubblewrap; sandbox-exec is coming).
    # Fall back to permissive so the worker doesn't fail-closed and strand the host.
    if [ "$OS" = "Darwin" ] && [ "$sandbox_policy" = "strict" ]; then
        warn "macOS strict sandbox (sandbox-exec) isn't available yet — running PERMISSIVE for now."
        sandbox_policy="permissive"
    fi
    info "Sandbox policy: ${sandbox_policy}"

    # M3: inference flavors can opt into on-demand model downloads (consent moment).
    local auto_acquire
    auto_acquire=$(resolve_auto_acquire "$flavor")
    [ -n "$auto_acquire" ] && info "Auto-acquire models: $([ "$auto_acquire" = "true" ] && echo "yes (on-demand)" || echo "no")"

    # ── Find Python ──────────────────────────────────────────────────

    local python
    python=$(find_python) || fail "Python >= 3.${MIN_PYTHON_MINOR} is required. Install it and retry ($([ "$OS" = "Darwin" ] && echo 'macOS: brew install python@3.12' || echo 'e.g. sudo apt install python3.12 python3.12-venv'))."
    info "Using $python ($($python --version 2>&1))"

    # ── Fetch release ────────────────────────────────────────────────

    info "Fetching release info from GitHub …"
    local release_json
    release_json=$(fetch_release "$requested_version")

    local tag
    tag=$(printf '%s' "$release_json" | grep -o '"tag_name" *: *"[^"]*"' | head -1 | sed 's/.*: *"//;s/"//')
    [ -n "$tag" ] || fail "could not parse tag_name from release JSON"
    local version="${tag#v}"
    info "Release: ${tag}"

    # ── Collect asset URLs ───────────────────────────────────────────

    local arch
    arch=$(detect_arch)
    info "Architecture: ${arch}"

    local deb_name="auspexai-worker_${version}_${arch}.deb"
    local whl_pattern="auspexai_worker-${version}-py3-none-any.whl"

    local deb_url="" whl_url=""

    # .deb is a Linux fast path only; on macOS we always take the pip/wheel route.
    if [ "$OS" = "Linux" ]; then
        deb_url=$(printf '%s' "$release_json" \
            | grep -o '"browser_download_url" *: *"[^"]*"' \
            | grep "$deb_name" \
            | head -1 \
            | sed 's/.*: *"//;s/"//') || true
    fi

    whl_url=$(printf '%s' "$release_json" \
        | grep -o '"browser_download_url" *: *"[^"]*"' \
        | grep "$whl_pattern" \
        | head -1 \
        | sed 's/.*: *"//;s/"//') || true

    # ── Install ──────────────────────────────────────────────────────

    local tmpdir
    tmpdir=$(mktemp -d)
    TMPDIR_CLEANUP="$tmpdir"

    if [ -n "$deb_url" ]; then
        # ── Fast path: .deb ──────────────────────────────────────────
        info "Found .deb for ${arch} — using package manager"
        info "Downloading ${deb_name} …"
        curl -fSL -o "${tmpdir}/${deb_name}" "$deb_url"

        info "Installing with apt …"
        sudo apt install -y "${tmpdir}/${deb_name}"
        info "Installed via .deb"
    elif [ -n "$whl_url" ]; then
        # ── Fallback: pip into /opt venv ─────────────────────────────
        info "No .deb for ${arch} — installing from wheel"
        info "Downloading ${whl_pattern} …"
        curl -fSL -o "${tmpdir}/${whl_pattern}" "$whl_url"

        # Ensure build deps for compiled wheels (cryptography, etc.). Linux only —
        # macOS installs prebuilt wheels from PyPI and ships venv with python.
        if [ "$OS" = "Linux" ]; then
            info "Checking build dependencies …"
            local build_deps_needed=()
            dpkg -s libffi-dev >/dev/null 2>&1 || build_deps_needed+=(libffi-dev)
            dpkg -s libssl-dev >/dev/null 2>&1 || build_deps_needed+=(libssl-dev)
            dpkg -s "${python}-venv" >/dev/null 2>&1 || build_deps_needed+=("${python}-venv")
            if [ ${#build_deps_needed[@]} -gt 0 ]; then
                info "Installing build dependencies: ${build_deps_needed[*]}"
                sudo apt install -y "${build_deps_needed[@]}"
            fi
        fi

        # Create or reuse venv
        if [ ! -d "${INSTALL_PREFIX}" ]; then
            info "Creating venv at ${INSTALL_PREFIX} …"
            sudo "$python" -m venv "${INSTALL_PREFIX}"
        fi

        # Wipe old package dir before reinstall — pip overlays without
        # cleaning, so stale .py files from the previous version survive.
        sudo rm -rf "${INSTALL_PREFIX}"/lib/python*/site-packages/auspexai_worker*

        info "Installing wheel (this may compile native extensions) …"
        sudo "${INSTALL_PREFIX}/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel 2>/dev/null
        sudo "${INSTALL_PREFIX}/bin/pip" install --no-cache-dir "${tmpdir}/${whl_pattern}"

        # Symlink CLI into PATH
        info "Creating CLI symlink …"
        sudo ln -sf "${INSTALL_PREFIX}/bin/auspexai-worker" /usr/local/bin/auspexai-worker

        # Install the service: a launchd LaunchAgent on macOS, a systemd user unit
        # on Linux (one installer, OS-aware — same command on both).
        if [ "$OS" = "Darwin" ]; then
            info "Installing launchd agent …"
            mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
            cat > "$HOME/Library/LaunchAgents/network.auspexai.worker.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>network.auspexai.worker</string>
  <key>ProgramArguments</key>
  <array>
    <string>${INSTALL_PREFIX}/bin/auspexai-worker</string>
    <string>daemon</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>${HOME}/Library/Logs/auspexai-worker.log</string>
  <key>StandardErrorPath</key><string>${HOME}/Library/Logs/auspexai-worker.log</string>
</dict>
</plist>
PLIST
            info "launchd agent written; the worker will auto-start on login."
        else
        info "Installing systemd user unit …"
        sudo mkdir -p "${SYSTEMD_UNIT_DIR}"
        sudo tee "${SYSTEMD_UNIT_DIR}/auspexai-worker.service" >/dev/null <<'UNIT'
[Unit]
Description=AuspexAI volunteer worker (Phase 1)
Documentation=https://github.com/auspexai/worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/opt/auspexai-worker/bin/auspexai-worker daemon
Restart=on-failure
RestartSec=10
PrivateTmp=true
NoNewPrivileges=true
; §41(a) STRICT resource caps: delegate a cgroup-v2 subtree so the daemon can
; create per-unit child cgroups (memory.max / pids.max) around the runner. Off →
; the daemon degrades to the rlimit floor. ProtectControlGroups stays unset (=no)
; so the delegated subtree is writable.
Delegate=yes

[Install]
WantedBy=default.target
UNIT
        fi

        # Install AppArmor profile if AppArmor is enabled at kernel level
        if [ -x /sbin/apparmor_parser ] && [ -d "${APPARMOR_DIR}" ] && \
           [ -f /sys/module/apparmor/parameters/enabled ] && \
           [ "$(cat /sys/module/apparmor/parameters/enabled 2>/dev/null)" = "Y" ]; then
            info "Installing AppArmor profile …"
            sudo tee "${APPARMOR_DIR}/auspexai-worker" >/dev/null <<'APPARMOR'
abi <abi/4.0>,

include <tunables/global>

profile auspexai-worker /opt/auspexai-worker/bin/auspexai-worker {
  include <abstractions/base>
  include <abstractions/python>
  include <abstractions/nameservice>
  include <abstractions/ssl_certs>

  /opt/auspexai-worker/** mr,
  /opt/auspexai-worker/bin/auspexai-worker rix,
  /opt/auspexai-worker/bin/auspexai-worker-runner rix,
  /opt/auspexai-worker/bin/python* rix,

  /usr/bin/python3* rix,
  /usr/lib/python3* r,
  /usr/lib/python3*/** mr,

  owner @{HOME}/.local/state/auspexai-worker/ rw,
  owner @{HOME}/.local/state/auspexai-worker/** rwk,
  owner @{HOME}/.local/share/auspexai-worker/ rw,
  owner @{HOME}/.local/share/auspexai-worker/** rwk,
  owner @{HOME}/.config/auspexai-worker/ r,
  owner @{HOME}/.config/auspexai-worker/** r,

  owner /run/user/*/auspexai-worker/** rwk,
  owner /run/user/*/auspexai-worker/ rw,

  /run/systemd/userdb/ r,
  /run/systemd/userdb/** r,

  /sys/fs/cgroup/ r,
  /sys/fs/cgroup/** rw,

  /proc/*/stat r,
  /proc/*/cgroup r,

  /etc/auspexai-worker/ r,
  /etc/auspexai-worker/** r,
  /etc/machine-id r,
  /etc/os-release r,

  dbus send
       bus=session
       path=/org/freedesktop/secrets/**
       interface=org.freedesktop.Secret.*
       peer=(label=unconfined),
  dbus receive
       bus=session
       interface=org.freedesktop.Secret.*
       peer=(label=unconfined),
  dbus send
       bus=session
       path=/org/freedesktop/DBus
       interface=org.freedesktop.DBus
       member={Hello,AddMatch,RemoveMatch,GetNameOwner,NameHasOwner,StartServiceByName},
  /run/user/*/bus rw,

  network inet stream,
  network inet6 stream,
  network unix stream,
  network unix dgram,

  signal (send) set=(term, kill) peer=auspexai-worker//bwrap_sandbox,
  signal (receive) set=(term, kill, int, hup),

  /usr/bin/bwrap cx -> bwrap_sandbox,

  profile bwrap_sandbox flags=(unconfined) {
    userns,
    /usr/bin/bwrap mr,
    /opt/auspexai-worker/bin/auspexai-worker-runner rix,
    owner @{HOME}/.local/share/auspexai-worker/workspaces/ rw,
    owner @{HOME}/.local/share/auspexai-worker/workspaces/** rwlk,
    /tmp/auspexai-worker-*/ rw,
    /tmp/auspexai-worker-*/** rwlk,
  }
}
APPARMOR
            if /sbin/apparmor_parser -r -W "${APPARMOR_DIR}/auspexai-worker" >/dev/null 2>&1; then
                info "AppArmor profile loaded"
            else
                warn "AppArmor profile reload failed — see README for workarounds"
            fi
        fi

        # Reload systemd
        if command -v systemctl >/dev/null 2>&1; then
            sudo systemctl daemon-reload || true
        fi

        info "Installed via pip (wheel)"
    else
        fail "no .deb or .whl found in release ${tag} — check https://github.com/${GITHUB_REPO}/releases"
    fi

    # ── Verify ───────────────────────────────────────────────────────

    if [ -x "${INSTALL_PREFIX}/bin/auspexai-worker" ]; then
        local installed_version
        installed_version=$("${INSTALL_PREFIX}/bin/auspexai-worker" --version 2>/dev/null || echo "unknown")
        info "Installed: ${installed_version}"
    fi

    # ── Check runtime deps ───────────────────────────────────────────

    # §41(a) STRICT sandbox runtime deps. bubblewrap builds the sandbox;
    # libseccomp (loaded by pyseccomp via ctypes) is the syscall-denylist gate.
    # Both are required for STRICT — install them actively when the volunteer
    # chose strict (the consent already happened), warn-only otherwise.
    ensure_sandbox_deps "$sandbox_policy"

    # ── Enable linger so user service survives logout ─────────────

    if command -v loginctl >/dev/null 2>&1; then
        local current_user
        current_user=$(id -un)
        if [ "$(loginctl show-user "$current_user" -p Linger --value 2>/dev/null)" != "yes" ]; then
            info "Enabling loginctl linger for ${current_user} (keeps worker running after logout) …"
            sudo loginctl enable-linger "$current_user" 2>/dev/null \
                || warn "could not enable linger; worker may stop when you log out"
        fi
    fi

    # ── Apply flavor (§9 #46) ────────────────────────────────────────
    # BEFORE bootstrap/start: [inference] backend is read at daemon start.

    if [ -x "${INSTALL_PREFIX}/bin/auspexai-worker" ]; then
        apply_flavor "$flavor"
    fi

    # §41: record the volunteer's sandbox-policy choice — surgical worker.toml
    # edit via the worker CLI, like the flavor. Guarded for binaries that
    # predate `sandbox set-policy` (< v0.2.16).
    if [ -x "${INSTALL_PREFIX}/bin/auspexai-worker" ] \
        && "${INSTALL_PREFIX}/bin/auspexai-worker" sandbox set-policy --help >/dev/null 2>&1; then
        "${INSTALL_PREFIX}/bin/auspexai-worker" sandbox set-policy "$sandbox_policy" >/dev/null \
            || warn "could not record [sandbox] policy in worker.toml"
    fi

    # M3: record the auto-acquire choice (inference flavors only) — surgical
    # [executor] auto_acquire write via the CLI, touching only the flag, not the
    # execution policy. Guarded for binaries predating `executor auto-acquire`
    # (< v0.2.21); on older ones the volunteer can set it from the dashboard.
    if [ -n "$auto_acquire" ] && [ -x "${INSTALL_PREFIX}/bin/auspexai-worker" ] \
        && "${INSTALL_PREFIX}/bin/auspexai-worker" executor auto-acquire --help >/dev/null 2>&1; then
        if "${INSTALL_PREFIX}/bin/auspexai-worker" executor auto-acquire \
            "$([ "$auto_acquire" = "true" ] && echo on || echo off)" >/dev/null; then
            info "Auto-acquire models: $([ "$auto_acquire" = "true" ] && echo "on (downloads on demand)" || echo "off (set-up models only)")"
        else
            warn "could not record [executor] auto_acquire in worker.toml"
        fi
    fi

    # ── Bootstrap + start ───────────────────────────────────────────

    if [ -x "${INSTALL_PREFIX}/bin/auspexai-worker" ]; then
        # Check if already enrolled
        local enrolled
        enrolled=$("${INSTALL_PREFIX}/bin/auspexai-worker" status 2>&1 | grep -c "worker-id:" || true)

        if [ "$enrolled" = "0" ]; then
            echo ""
            printf 'Bootstrap now? This generates a keypair and enrolls with the coordinator. [Y/n] '
            read -r reply </dev/tty
            case "$reply" in
                n|N|no|NO) ;;
                *)
                    info "Bootstrapping …"
                    "${INSTALL_PREFIX}/bin/auspexai-worker" bootstrap
                    info "Starting service …"
                    start_worker_service
                    ;;
            esac
        else
            info "Already enrolled — skipping bootstrap"
            echo ""
            printf 'Start the service now? [Y/n] '
            read -r reply </dev/tty
            case "$reply" in
                n|N|no|NO) ;;
                *)
                    info "Starting service …"
                    start_worker_service
                    ;;
            esac
        fi
    fi

    # ── Offer model setup (BYOM onramp, W-M) ─────────────────────────
    # Opt-in (default N) — never surprise a volunteer with multi-GB downloads.
    # The base install is lean; pulling models needs the huggingface_hub extra,
    # installed here only if the volunteer opts in.
    if [ -x "${INSTALL_PREFIX}/bin/auspexai-worker" ]; then
        echo ""
        printf 'Set up inference models now? Downloads models that fit your hardware so this worker can run real experiments. [y/N] '
        read -r reply </dev/tty
        case "$reply" in
            y|Y|yes|YES)
                if [ -x "${INSTALL_PREFIX}/bin/pip" ]; then
                    info "Installing model-download support (huggingface_hub) …"
                    sudo "${INSTALL_PREFIX}/bin/pip" install -q huggingface_hub \
                        || warn "could not install huggingface_hub; \`model pull\` will be unavailable"
                else
                    warn "pip not found in ${INSTALL_PREFIX}; install huggingface_hub manually for \`model pull\`"
                fi
                "${INSTALL_PREFIX}/bin/auspexai-worker" model setup </dev/tty || true
                ;;
            *)
                echo "    Skipped. Run \`auspexai-worker model recommend\` to see what fits,"
                echo "    then \`auspexai-worker model setup\` anytime."
                ;;
        esac
    fi

    cat <<'EOF'

Done. Useful commands:

  auspexai-worker status         # identity, tier, progress
  auspexai-worker logs -f        # watch daemon activity in real time
  auspexai-worker login          # optional: bind GitHub identity for T1 trust
  auspexai-worker model setup    # pick + download models that fit this host
  auspexai-worker model list     # models you have (your network inventory)

Your worker also has a local web dashboard (while the daemon is running):

  http://127.0.0.1:7799          # status, activity, receipts, models, settings
                                 # local-only; never exposed to the network

Changing flavor later: re-run this installer — the menu defaults to your
current choice (Enter keeps it) — or pass --flavor <name> explicitly.
See --list-flavors for what each provides.

EOF

    if [ -n "$FLAVOR_ISSUES" ]; then
        echo ""
        warn "FLAVOR SETUP ISSUES — the worker installed, but flavor '${flavor}' is incomplete:"
        # shellcheck disable=SC2059 — FLAVOR_ISSUES embeds \n separators by design
        printf "$FLAVOR_ISSUES" >&2
        warn "Re-running this installer (same flavor) retries these steps."
    fi
}

main "$@"
