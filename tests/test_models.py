"""W-M — model catalog, store, recommend, and pull (fetcher-mocked)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from auspexai_worker.models import (
    FileCatalogSource,
    ModelCatalog,
    ModelStore,
    WorkerResources,
    recommend,
)
from auspexai_worker.models.catalog import ModelCatalogEntry
from auspexai_worker.models.fetch import HfHubFetcher, ModelFetchError, pull_model

# ---- catalog ---------------------------------------------------------------


def test_bundled_seed_catalog_loads():
    catalog = ModelCatalog.load()
    assert len(catalog) >= 3
    gpt2 = catalog.get("gpt2")
    assert gpt2 is not None
    assert gpt2.hf_repo == "openai-community/gpt2"
    assert gpt2.disk_bytes > 0


def test_file_catalog_source(tmp_path: Path):
    p = tmp_path / "cat.json"
    p.write_text('[{"id": "m", "hf_repo": "x/y", "disk_bytes": 10}]', encoding="utf-8")
    catalog = ModelCatalog.load(FileCatalogSource(p))
    assert len(catalog) == 1
    assert catalog.get("m").hf_repo == "x/y"


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


# ---- recommend -------------------------------------------------------------

SMALL = ModelCatalogEntry(id="small", version="1", hf_repo="a/b", disk_bytes=10, min_ram_gb=1)
BIG = ModelCatalogEntry(
    id="big", version="1", hf_repo="c/d", disk_bytes=15_000_000_000, vram_load_gb=16, min_ram_gb=32
)


def test_recommend_fit_and_blockers(tmp_path: Path):
    catalog = ModelCatalog([SMALL, BIG])
    store = ModelStore(tmp_path)
    res = WorkerResources(disk_free_bytes=1_000_000_000, ram_gb=8.0, vram_gb=None)
    recs = {r.entry.id: r for r in recommend(catalog, store, res)}
    assert recs["small"].fits is True
    assert recs["big"].fits is False
    # big is blocked on both disk and RAM
    assert any("disk" in b for b in recs["big"].blockers)
    assert any("RAM" in b for b in recs["big"].blockers)
    # vram is a soft note, not a blocker
    assert recs["big"].notes


def test_recommend_demand_sorts_first(tmp_path: Path):
    catalog = ModelCatalog([SMALL, BIG])
    store = ModelStore(tmp_path)
    res = WorkerResources(disk_free_bytes=100_000_000_000, ram_gb=64.0, vram_gb=80.0)
    recs = recommend(catalog, store, res, demand=("big",))
    assert recs[0].entry.id == "big"  # in-demand floats to the top
    assert recs[0].in_demand is True


def test_recommend_marks_installed(tmp_path: Path):
    _stage_model(tmp_path, "small")
    recs = {
        r.entry.id: r
        for r in recommend(
            ModelCatalog([SMALL]), ModelStore(tmp_path), WorkerResources(10**12, 64, 80)
        )
    }
    assert recs["small"].installed is True


# ---- pull (fetcher-mocked) -------------------------------------------------


class _FakeFetcher:
    def __init__(self, content: bytes = b"weights", fname: str = "model.bin") -> None:
        self.content = content
        self.fname = fname
        self.calls = 0

    def fetch(self, entry, dest_dir: Path) -> None:
        self.calls += 1
        (dest_dir / self.fname).write_bytes(self.content)


def test_pull_installs_and_is_idempotent(tmp_path: Path):
    store = ModelStore(tmp_path)
    fetcher = _FakeFetcher()
    dest = pull_model(SMALL, store, fetcher)
    assert dest == tmp_path / "small"
    assert (dest / "model.bin").read_bytes() == b"weights"
    assert store.has("small")
    # second pull is a no-op (already installed)
    pull_model(SMALL, store, fetcher)
    assert fetcher.calls == 1


def test_pull_verifies_sha256(tmp_path: Path):
    content = b"trusted-weights"
    good_sha = hashlib.sha256(content).hexdigest()
    entry = ModelCatalogEntry(
        id="m", version="1", hf_repo="a/b", disk_bytes=10, sha256={"model.bin": good_sha}
    )
    store = ModelStore(tmp_path)
    pull_model(entry, store, _FakeFetcher(content=content))
    assert store.has("m")


def test_pull_rejects_sha256_mismatch_and_cleans_up(tmp_path: Path):
    entry = ModelCatalogEntry(
        id="m", version="1", hf_repo="a/b", disk_bytes=10, sha256={"model.bin": "deadbeef"}
    )
    store = ModelStore(tmp_path)
    with pytest.raises(ModelFetchError, match="integrity"):
        pull_model(entry, store, _FakeFetcher(content=b"tampered"))
    assert not store.has("m")  # not installed
    assert not (tmp_path / "m.partial").exists()  # staging cleaned


def test_pull_disk_precheck(tmp_path: Path):
    store = ModelStore(tmp_path)
    res = WorkerResources(disk_free_bytes=5, ram_gb=64, vram_gb=None)
    with pytest.raises(ModelFetchError, match="insufficient disk"):
        pull_model(BIG, store, _FakeFetcher(), resources=res)


def test_hf_fetcher_missing_dependency(tmp_path: Path):
    # huggingface_hub isn't installed in the test env -> clear actionable error.
    with pytest.raises(ModelFetchError, match="huggingface_hub"):
        HfHubFetcher().fetch(SMALL, tmp_path)


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


def _cat_file(tmp_path: Path) -> Path:
    p = tmp_path / "cat.json"
    p.write_text(
        '[{"id": "tiny", "hf_repo": "a/b", "disk_bytes": 10, "note": "tiny"}]', encoding="utf-8"
    )
    return p


def test_cli_model_recommend_and_list(tmp_path: Path):
    from click.testing import CliRunner

    from auspexai_worker.cli import cli

    env = {"AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data")}
    runner = CliRunner()
    r = runner.invoke(cli, ["model", "recommend", "--catalog", str(_cat_file(tmp_path))], env=env)
    assert r.exit_code == 0, r.output
    assert "tiny" in r.output and "fits" in r.output
    r2 = runner.invoke(cli, ["model", "list"], env=env)
    assert r2.exit_code == 0
    assert "no models" in r2.output


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


def test_cli_model_setup_non_interactive_pulls_nothing(tmp_path: Path):
    from click.testing import CliRunner

    from auspexai_worker.cli import cli

    env = {"AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data")}
    # CliRunner provides a non-tty stdin -> setup must not download, just guide.
    r = CliRunner().invoke(cli, ["model", "setup", "--catalog", str(_cat_file(tmp_path))], env=env)
    assert r.exit_code == 0, r.output
    assert "Non-interactive" in r.output
    assert not (tmp_path / "data" / "models" / "tiny").exists()
