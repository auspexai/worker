"""W-M — model store, coords-pull, and selection parsing (fetcher-mocked).

The network's provisionable-model catalog now lives on the coordinator
(`GET /api/v0/models/supported`); the worker's static seed catalog + the
catalog-based `recommend()` / `pull_model()` were retired. What remains local:
the store, the HF-direct pull utilities, and selection parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auspexai_worker.models import ModelStore
from auspexai_worker.models.fetch import (
    HfHubFetcher,
    ModelFetchError,
    StoreModelAcquirer,
    pull_from_coords,
)

# ---- store -----------------------------------------------------------------


def _stage_model(root: Path, model_id: str, *, fname: str = "w.bin", content: bytes = b"W") -> None:
    d = root / model_id
    d.mkdir(parents=True)
    (d / fname).write_bytes(content)


def test_store_list_has_inventory_rm(tmp_path: Path):
    store = ModelStore(tmp_path)
    assert store.list() == []
    assert not store.has("gpt2")
    _stage_model(tmp_path, "gpt2")
    assert store.has("gpt2")
    assert store.inventory() == ["gpt2"]
    assert store.remove("gpt2") is True
    assert not store.has("gpt2")
    assert store.remove("gpt2") is False  # already gone


def test_store_empty_dir_is_not_installed(tmp_path: Path):
    (tmp_path / "half-pull").mkdir()
    store = ModelStore(tmp_path)
    assert not store.has("half-pull")  # bare dir doesn't count
    assert store.list() == []


# ---- HF fetcher: missing optional dependency -------------------------------


def test_hf_fetcher_missing_dependency(tmp_path: Path):
    # huggingface_hub isn't installed in the test env -> clear actionable error.
    with pytest.raises(ModelFetchError, match="huggingface_hub"):
        HfHubFetcher().fetch_file("a/b", "model.gguf", tmp_path)


# ---- M3 lazy auto-acquire: pull_from_coords + StoreModelAcquirer -----------


class _FakeFileFetcher:
    """Fetcher exposing the `fetch_file(repo, filename, dest_dir)` surface the
    coords-pull path uses."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._raises = raises

    def fetch_file(self, repo: str, filename: str, dest_dir: Path) -> None:
        self.calls.append((repo, filename))
        if self._raises is not None:
            raise self._raises
        (dest_dir / filename).write_bytes(b"weights")


def test_pull_from_coords_installs_and_idempotent(tmp_path: Path):
    store = ModelStore(tmp_path)
    fetcher = _FakeFileFetcher()
    dest = pull_from_coords(
        model_id="m-x", hf_repo="Org/M-GGUF", hf_filename="M-Q4.gguf", store=store, fetcher=fetcher
    )
    assert dest == tmp_path / "m-x"
    assert (dest / "M-Q4.gguf").read_bytes() == b"weights"
    assert store.has("m-x")
    # already installed -> no-op (fetcher not called again)
    pull_from_coords(
        model_id="m-x", hf_repo="Org/M-GGUF", hf_filename="M-Q4.gguf", store=store, fetcher=fetcher
    )
    assert len(fetcher.calls) == 1


def test_pull_from_coords_headroom_guard(tmp_path: Path):
    store = ModelStore(tmp_path)
    with pytest.raises(ModelFetchError, match="headroom"):
        pull_from_coords(
            model_id="m-x",
            hf_repo="Org/M-GGUF",
            hf_filename="M-Q4.gguf",
            store=store,
            fetcher=_FakeFileFetcher(),
            disk_free_bytes=1_000_000,  # below the 2 GB default min headroom
        )


def test_pull_from_coords_cleans_staging_on_failure(tmp_path: Path):
    store = ModelStore(tmp_path)
    with pytest.raises(ModelFetchError, match="auto-acquire"):
        pull_from_coords(
            model_id="m-x",
            hf_repo="Org/M-GGUF",
            hf_filename="M-Q4.gguf",
            store=store,
            fetcher=_FakeFileFetcher(raises=RuntimeError("connection died")),
        )
    assert not store.has("m-x")
    assert not (tmp_path / "m-x.partial").exists()  # staging cleaned


def test_store_model_acquirer_pulls_into_store(tmp_path: Path):
    store = ModelStore(tmp_path)
    acquirer = StoreModelAcquirer(store, _FakeFileFetcher())
    dest = acquirer.acquire(model_id="m-x", hf_repo="Org/M-GGUF", hf_filename="M-Q4.gguf")
    assert dest == tmp_path / "m-x"
    assert store.has("m-x")


# ---- selection parsing (interactive setup) ---------------------------------


def test_parse_selection():
    from auspexai_worker.models.recommend import parse_selection

    assert parse_selection("all", 3) == [0, 1, 2]
    assert parse_selection("*", 3) == [0, 1, 2]
    assert parse_selection("none", 3) == []
    assert parse_selection("", 3) == []
    assert parse_selection("1,3", 3) == [0, 2]
    assert parse_selection("1 2", 3) == [0, 1]
    assert parse_selection("2, 2, 9, foo", 3) == [1]  # dedup + drop OOR + garbage


# ---- CLI ($AUSPEXAI_WORKER_DATA_DIR drives the store) ----------------------


def test_cli_model_recommend_uses_network_catalog(tmp_path: Path, monkeypatch):
    """When the worker can reach the coordinator, `model recommend` lists the
    network catalog entries that fit this host's memory budget."""
    from click.testing import CliRunner

    from auspexai_worker import cli as cli_module
    from auspexai_worker.cli import cli
    from auspexai_worker.coordinator import SupportedModel

    def _fake_supported(_config):
        return [
            SupportedModel(
                model_id="qwen2.5-0.5b-instruct-q4_k_m",
                display_name="Qwen2.5 0.5B",
                approx_ram_gb=1.5,
                status="available",
                hf_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                param_b=0.5,
            ),
            SupportedModel(
                model_id="way-too-big",
                display_name="Huge",
                approx_ram_gb=9_999.0,  # never fits any real host budget
                status="runnable",
                hf_repo="Org/Huge-GGUF",
                param_b=999.0,
            ),
        ]

    monkeypatch.setattr(cli_module, "_network_supported_models", _fake_supported)
    env = {"AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data")}
    r = CliRunner().invoke(cli, ["model", "recommend"], env=env)
    assert r.exit_code == 0, r.output
    assert "from the network catalog fit this host" in r.output
    assert "qwen2.5-0.5b-instruct-q4_k_m" in r.output
    assert "way-too-big" not in r.output  # filtered out by budget


def test_cli_model_recommend_offline_fallback(tmp_path: Path):
    """Unenrolled worker (no coordinator credential) + no huggingface_hub ->
    falls back to the direct-HF path and degrades gracefully."""
    from click.testing import CliRunner

    from auspexai_worker.cli import cli

    env = {"AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data")}
    r = CliRunner().invoke(cli, ["model", "recommend"], env=env)
    assert r.exit_code == 0, r.output
    assert "direct-HuggingFace fallback" in r.output
    assert "HuggingFace unavailable" in r.output


def test_cli_model_list_empty(tmp_path: Path):
    from click.testing import CliRunner

    from auspexai_worker.cli import cli

    env = {"AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data")}
    r = CliRunner().invoke(cli, ["model", "list"], env=env)
    assert r.exit_code == 0
    assert "no models" in r.output


def test_capabilities_declares_model_inventory():
    from auspexai_worker.capabilities import collect

    d = collect(models=["gpt2", "all-minilm-l6-v2"]).to_dict()
    assert d["models"] == ["gpt2", "all-minilm-l6-v2"]
    # empty inventory is omitted from the wire (compact heartbeat)
    assert "models" not in collect().to_dict()


def test_capabilities_declares_worker_version():
    from auspexai_worker import __version__
    from auspexai_worker.capabilities import collect

    # version is ALWAYS reported (mixed-fleet feature detection by the operator).
    assert collect().to_dict()["worker_version"] == __version__


def test_cli_model_setup_offline_is_graceful(tmp_path: Path):
    from click.testing import CliRunner

    from auspexai_worker.cli import cli

    env = {"AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data")}
    # No huggingface_hub in the test env -> HF-direct setup degrades gracefully
    # and pulls nothing.
    r = CliRunner().invoke(cli, ["model", "setup"], env=env)
    assert r.exit_code == 0, r.output
    assert "HuggingFace unavailable" in r.output
    models_dir = tmp_path / "data" / "models"
    assert not models_dir.exists() or not any(models_dir.iterdir())


def test_cli_model_setup_shows_installed_inventory(tmp_path: Path):
    # #6: the setup/upgrade path must REPORT what's already on the device
    # (preserved across upgrades, not re-downloaded), so a re-run isn't mistaken
    # for redundant downloading. The summary is local-only, so it shows even when
    # HuggingFace is unavailable (as in the test env).
    from click.testing import CliRunner

    from auspexai_worker.cli import cli

    data = tmp_path / "data"
    _stage_model(data / "models", "qwen3-q4", content=b"x" * 100)
    env = {"AUSPEXAI_WORKER_DATA_DIR": str(data)}
    r = CliRunner().invoke(cli, ["model", "setup"], env=env)
    assert r.exit_code == 0, r.output
    assert "already available on this device" in r.output
    assert "qwen3-q4" in r.output
    assert "not re-downloaded" in r.output


def test_pull_quant_installs_single_gguf(tmp_path: Path):
    from auspexai_worker.models.fetch import pull_quant
    from auspexai_worker.models.hf_browse import ModelQuant

    class _FileFetcher:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_file(self, repo: str, filename: str, dest_dir: Path) -> None:
            self.calls += 1
            (dest_dir / filename).write_bytes(b"gguf-bytes")

    q = ModelQuant(repo="org/M-GGUF", filename="M-Q4_K_M.gguf", quant="Q4_K_M", size_bytes=9)
    store = ModelStore(tmp_path)
    fetcher = _FileFetcher()
    dest = pull_quant(q, store, fetcher)
    assert store.has(q.model_id)
    assert (dest / "M-Q4_K_M.gguf").read_bytes() == b"gguf-bytes"
    pull_quant(q, store, fetcher)  # idempotent
    assert fetcher.calls == 1
