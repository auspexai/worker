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

# ── Helpers ──────────────────────────────────────────────────────────

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m==> WARNING:\033[0m %s\n' "$*" >&2; }
fail()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "'$1' is required but not found. Install it and retry."
}

# ── Detect Python ────────────────────────────────────────────────────

find_python() {
    for candidate in python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            local ver
            ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.minor}")' 2>/dev/null) || continue
            if [ "$ver" -ge "$MIN_PYTHON_MINOR" ] 2>/dev/null; then
                echo "$candidate"
                return
            fi
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

main() {
    local requested_version=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --version) requested_version="$2"; shift 2 ;;
            --uninstall)
                do_uninstall
                exit 0
                ;;
            --help|-h)
                echo "Usage: install.sh [--version VERSION] [--uninstall]"
                echo ""
                echo "Installs the AuspexAI worker from the latest GitHub release."
                echo "If a .deb exists for this architecture, it is preferred."
                echo "Otherwise, a pip-based install into /opt/auspexai-worker/ is used."
                echo ""
                echo "  --uninstall   Stop, de-enroll, and remove the worker"
                echo "  --version V   Install a specific version instead of latest"
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

    # ── Find Python ──────────────────────────────────────────────────

    local python
    python=$(find_python) || fail "Python >= 3.${MIN_PYTHON_MINOR} is required. Install python3.11 or python3.12 and retry."
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

    deb_url=$(printf '%s' "$release_json" \
        | grep -o '"browser_download_url" *: *"[^"]*"' \
        | grep "$deb_name" \
        | head -1 \
        | sed 's/.*: *"//;s/"//') || true

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

        # Ensure build deps for compiled wheels (cryptography, etc.)
        info "Checking build dependencies …"
        local build_deps_needed=()
        dpkg -s libffi-dev >/dev/null 2>&1 || build_deps_needed+=(libffi-dev)
        dpkg -s libssl-dev >/dev/null 2>&1 || build_deps_needed+=(libssl-dev)
        dpkg -s "${python}-venv" >/dev/null 2>&1 || build_deps_needed+=("${python}-venv")
        if [ ${#build_deps_needed[@]} -gt 0 ]; then
            info "Installing build dependencies: ${build_deps_needed[*]}"
            sudo apt install -y "${build_deps_needed[@]}"
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
        sudo "${INSTALL_PREFIX}/bin/pip" install --upgrade pip setuptools wheel 2>/dev/null
        sudo "${INSTALL_PREFIX}/bin/pip" install "${tmpdir}/${whl_pattern}"

        # Symlink CLI into PATH
        info "Creating CLI symlink …"
        sudo ln -sf "${INSTALL_PREFIX}/bin/auspexai-worker" /usr/local/bin/auspexai-worker

        # Install systemd user unit
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

[Install]
WantedBy=default.target
UNIT

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

    if ! command -v bwrap >/dev/null 2>&1; then
        warn "bubblewrap (bwrap) is not installed — sandbox isolation won't work"
        echo "    Install it with: sudo apt install bubblewrap"
    fi

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
                    if ! systemctl --user enable --now auspexai-worker.service 2>/dev/null; then
                        info "systemd user service unavailable; starting daemon directly …"
                        nohup "${INSTALL_PREFIX}/bin/auspexai-worker" daemon </dev/null >/dev/null 2>&1 &
                        info "daemon started (pid $!); logs at: auspexai-worker logs -f"
                    fi
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
                    if ! systemctl --user enable --now auspexai-worker.service 2>/dev/null; then
                        info "systemd user service unavailable; starting daemon directly …"
                        nohup "${INSTALL_PREFIX}/bin/auspexai-worker" daemon </dev/null >/dev/null 2>&1 &
                        info "daemon started (pid $!); logs at: auspexai-worker logs -f"
                    fi
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

EOF
}

main "$@"
