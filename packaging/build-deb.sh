#!/usr/bin/env bash
# Build the auspexai-worker .deb inside a fresh Ubuntu 24.04 container.
#
# Why containerized: keeps the host clean of debhelper / dh-virtualenv /
# devscripts build deps, AND matches the environment volunteer users
# install onto (Ubuntu 24.04+ Phase 1 target). Same image is used by the
# M7c GitHub Actions release workflow.
#
# Usage:
#   packaging/build-deb.sh              # build only
#   packaging/build-deb.sh --test       # build, then install-test in a
#                                       # second clean container
#
# Output:
#   /tmp/auspexai-deb-build/auspexai-worker_<version>_amd64.deb
#   /tmp/auspexai-deb-build/auspexai-worker_<version>_amd64.{changes,buildinfo}
#
# Requirements: podman (or docker — the script auto-detects).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRATCH_DIR="${AUSPEXAI_DEB_SCRATCH:-/tmp/auspexai-deb-build}"
IMAGE="ubuntu:24.04"

if command -v podman >/dev/null 2>&1; then
    RUNTIME=podman
elif command -v docker >/dev/null 2>&1; then
    RUNTIME=docker
else
    echo "build-deb.sh: ERROR — neither podman nor docker found on PATH" >&2
    exit 1
fi

RUN_TESTS=0
for arg in "$@"; do
    case "$arg" in
        --test)  RUN_TESTS=1 ;;
        --help|-h)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)  echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

echo "build-deb.sh: staging source → $SCRATCH_DIR/worker"
rm -rf "$SCRATCH_DIR/worker"
mkdir -p "$SCRATCH_DIR"
tar --exclude=.venv --exclude=__pycache__ --exclude='*.pyc' \
    --exclude='.pytest_cache' --exclude='*.egg-info' \
    -cf - -C "$(dirname "$REPO_ROOT")" "$(basename "$REPO_ROOT")" | \
    tar -xf - -C "$SCRATCH_DIR/"
# Rename to a stable name regardless of repo-dir name
if [ "$(basename "$REPO_ROOT")" != "worker" ]; then
    mv "$SCRATCH_DIR/$(basename "$REPO_ROOT")" "$SCRATCH_DIR/worker"
fi

echo "build-deb.sh: building .deb inside $IMAGE container via $RUNTIME"
$RUNTIME run --rm \
  -v "$SCRATCH_DIR:/build:Z" \
  -w /build/worker \
  "$IMAGE" \
  bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends \
      dh-virtualenv debhelper devscripts dpkg-dev \
      python3-venv python3-dev python3.12-venv python3.12-dev \
      python3-pip build-essential ca-certificates 2>&1 | tail -3
    dpkg-buildpackage -us -uc -b
  '

echo ""
echo "build-deb.sh: build complete. Artifacts:"
ls -lh "$SCRATCH_DIR"/auspexai-worker_*.deb \
       "$SCRATCH_DIR"/auspexai-worker_*.buildinfo \
       "$SCRATCH_DIR"/auspexai-worker_*.changes 2>/dev/null

if [ "$RUN_TESTS" -eq 1 ]; then
    DEB_FILE="$(ls "$SCRATCH_DIR"/auspexai-worker_*.deb | head -1)"
    DEB_NAME="$(basename "$DEB_FILE")"
    echo ""
    echo "build-deb.sh: install-testing $DEB_NAME in a clean $IMAGE container"
    $RUNTIME run --rm \
      -v "$SCRATCH_DIR:/build:Z" \
      "$IMAGE" \
      bash -c "
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y --no-install-recommends /build/$DEB_NAME 2>&1 | tail -25
        echo ''
        echo '=== verify installed files ==='
        ls /opt/auspexai-worker/bin/auspexai-worker /opt/auspexai-worker/bin/auspexai-worker-runner
        ls /etc/systemd/user/auspexai-worker.service /etc/apparmor.d/auspexai-worker
        echo ''
        echo '=== smoke test ==='
        /opt/auspexai-worker/bin/auspexai-worker --version
      "
fi
