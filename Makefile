# Local CI gate — mirrors the test job of .github/workflows/ci.yml
# step-for-step so a green `make ci` is a green CI run (CI additionally runs
# the same steps on a 3.11/3.12 matrix, plus the independent build-deb job).
# Run before every push.
#
# Dev env is `--extra dev` ONLY: --all-extras would pull optional runtime
# extras that some tests assert are absent.
.PHONY: ci sync lint test build

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
