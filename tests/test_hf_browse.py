"""HF-direct, footprint-filtered model evaluation (v0.1.11)."""

from __future__ import annotations

from auspexai_worker.models.hf_browse import (
    ModelQuant,
    parse_quant,
    quant_fits,
    runnable_models,
    usable_budget_gb,
)


def _q(repo: str, quant: str, gb: float) -> ModelQuant:
    return ModelQuant(repo=repo, filename=f"x-{quant}.gguf", quant=quant, size_bytes=int(gb * 1e9))


class _FakeBrowser:
    def __init__(self, data: dict[str, list[ModelQuant]]):
        self._data = data

    def search(self, *, limit: int) -> list[str]:
        return list(self._data)[:limit]

    def quants(self, repo: str) -> list[ModelQuant]:
        return self._data[repo]


def test_parse_quant():
    assert parse_quant("Llama-3.2-3B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert parse_quant("model-IQ4_XS.gguf") == "IQ4_XS"
    assert parse_quant("foo-Q8_0.gguf") == "Q8_0"
    assert parse_quant("bar.f16.gguf") == "F16"


def test_usable_budget():
    assert usable_budget_gb(8.0, unified=True) == 5.0  # 3 GB OS headroom (unified reality)
    assert usable_budget_gb(24.0, unified=False) == 23.5  # 0.5 GB discrete headroom
    assert usable_budget_gb(None, unified=True) is None
    assert usable_budget_gb(1.0, unified=True) == 0.0  # floored


def test_quant_fits():
    big_disk = 10**12
    assert quant_fits(_q("r", "Q4_K_M", 2.0), 5.4, big_disk) is True  # 2*1.2=2.4 <= 5.4
    assert quant_fits(_q("r", "Q8_0", 6.0), 5.4, big_disk) is False  # 6*1.2=7.2 > 5.4
    assert quant_fits(_q("r", "Q4_K_M", 2.0), 5.4, 1_000) is False  # disk too small
    assert quant_fits(_q("r", "Q4_K_M", 2.0), None, big_disk) is True  # unknown budget -> disk only


def test_runnable_models_picks_largest_fitting():
    data = {
        # repo with two quants; on a 7.4 GB unified host (usable 5.4) Q8 is too big.
        "org/llama-3b": [_q("org/llama-3b", "Q4_K_M", 2.0), _q("org/llama-3b", "Q8_0", 8.0)],
        # all quants too big -> excluded entirely.
        "org/huge-70b": [_q("org/huge-70b", "Q4_K_M", 40.0)],
        # small model, multiple fit -> largest-fitting wins (best quality that runs).
        "org/mini": [_q("org/mini", "Q4_K_M", 0.5), _q("org/mini", "Q8_0", 1.0)],
    }
    out = runnable_models(
        _FakeBrowser(data),
        memory_budget_gb=7.4,
        unified=True,
        disk_free_bytes=10**12,
    )
    picked = {r.quant.repo: r.quant.quant for r in out}
    assert picked == {"org/llama-3b": "Q4_K_M", "org/mini": "Q8_0"}  # 70b excluded
    assert "org/huge-70b" not in picked


def test_runnable_models_marks_installed():
    data = {"org/m": [_q("org/m", "Q4_K_M", 1.0)]}
    inst = frozenset({ModelQuant("org/m", "x-Q4_K_M.gguf", "Q4_K_M", 10**9).model_id})
    out = runnable_models(
        _FakeBrowser(data),
        memory_budget_gb=16.0,
        unified=False,
        disk_free_bytes=10**12,
        installed_ids=inst,
    )
    assert out[0].installed is True


def test_model_id_is_filesystem_safe():
    q = _q("bartowski/Llama-3.2-3B-Instruct-GGUF", "Q4_K_M", 2.0)
    assert q.model_id == "llama-3.2-3b-instruct-gguf-q4_k_m"
    assert "/" not in q.model_id
