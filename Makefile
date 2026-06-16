# Local CI gate — mirrors the test job of .github/workflows/ci.yml
# step-for-step so a green `make ci` is a green CI run (CI additionally runs
# the same steps on a 3.11/3.12 matrix, plus the independent build-deb job).
# Run before every push.
#
# Dev env is `--extra dev` ONLY: --all-extras would pull optional runtime
# extras that some tests assert are absent.
.PHONY: ci sync lint test build redteam

ci: sync lint test build

sync:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -v

build:
	uv build

# Full §41(a) STRICT escape proof — all five vectors, INCLUDING the cgroup
# memory + fork-bomb caps, which only fire under a delegated cgroup subtree.
# We run the red-team module under `systemd-run --user -p Delegate=yes` so the
# resource vectors actually execute, and set AUSPEXAI_REDTEAM_REQUIRE_FULL=1 so
# a missing precondition (no delegation / no bwrap) FAILS LOUDLY instead of
# skipping. Requires a systemd user manager (loginctl enable-linger), bubblewrap
# and libseccomp — i.e. a worker-like host or the self-hosted `sandbox-redteam`
# CI runner, NOT a hosted GitHub runner (no cgroup delegation there).
redteam: sync
	systemd-run --user --wait --pipe --quiet \
	  -p Delegate=yes \
	  --working-directory="$(CURDIR)" \
	  -E PYTHONPATH="$(CURDIR)/src" \
	  -E AUSPEXAI_REDTEAM_REQUIRE_FULL=1 \
	  -- "$(CURDIR)/.venv/bin/python3" -m pytest tests/test_redteam_strict.py -v
