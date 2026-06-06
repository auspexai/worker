"""HF-direct model evaluation — what *this* worker can run, from HuggingFace.

Per the reframed W-M model (worker_model_management_design.md §3a): the worker
evaluates **available models direct from HF** by its own resource footprint —
there is no internal catalog gating its choices. It queries the Hub for GGUF
text-generation models (real per-quant file sizes) and keeps the quants that fit
its accelerator-memory budget.

GGUF is the lever: each quant is a single file whose **size ≈ the memory to load
it** (+ a small runtime/KV overhead). So per-quant fit falls straight out of HF
data — no guessed footprints. The network-facing "catalog" is the bottom-up
aggregate of what workers report (coordinator-side, layer 2); this is layer 1.

The HF access is behind `HfBrowser` so the logic is tested with a fake; the real
`HfHubBrowser` lazily imports the optional `huggingface_hub`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

# Memory overhead beyond the weight file (KV cache + runtime), and headroom left
# for the OS — larger on unified memory where the accelerator shares system RAM.
_LOAD_OVERHEAD = 1.2
_UNIFIED_HEADROOM_GB = 2.0
_DISCRETE_HEADROOM_GB = 0.5

_QUANT_RE = re.compile(r"(I?Q\d+(?:_[A-Za-z0-9]+)*|F16|BF16|F32)", re.IGNORECASE)


@dataclass(frozen=True)
class ModelQuant:
    """One downloadable quantization (a GGUF file on HF)."""

    repo: str  # HF repo id
    filename: str  # the .gguf file
    quant: str  # e.g. "Q4_K_M"
    size_bytes: int  # real file size from HF

    @property
    def model_id(self) -> str:
        """Store key: `<repo-slug>-<quant>` (lowercased), keeping it filesystem-safe."""
        slug = self.repo.split("/")[-1].lower()
        return f"{slug}-{self.quant.lower()}"

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9


def parse_quant(filename: str) -> str:
    m = _QUANT_RE.search(filename)
    return m.group(0).upper() if m else filename.rsplit(".", 1)[0]


class HfBrowser(Protocol):
    def search(self, *, limit: int) -> list[str]:
        """Return candidate HF repo ids (GGUF text-generation, popular first)."""
        ...

    def quants(self, repo: str) -> list[ModelQuant]:
        """Return the GGUF quant files (with real sizes) for a repo."""
        ...


class HfHubBrowser:
    """Real browser over the HuggingFace Hub (lazy `huggingface_hub` import)."""

    def _api(self):
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - exercised via the extra
            raise RuntimeError(
                "huggingface_hub not installed; `pip install 'auspexai-worker[models]'`"
            ) from exc
        return HfApi()

    def search(self, *, limit: int) -> list[str]:
        # huggingface_hub.HfApi.list_models verified params: filter / sort /
        # limit (no `task`, no `direction`; `sort="downloads"` is most-first).
        # `filter="gguf"` selects GGUF-tagged repos; the budget filter narrows
        # to what runs. We don't over-filter on pipeline_tag (many GGUF
        # conversion repos omit it, which would hide good models).
        models = self._api().list_models(filter="gguf", sort="downloads", limit=limit)
        return [m.id for m in models]

    def quants(self, repo: str) -> list[ModelQuant]:
        info = self._api().model_info(repo, files_metadata=True)
        out: list[ModelQuant] = []
        for sib in info.siblings or []:
            name = sib.rfilename
            if not name.lower().endswith(".gguf") or sib.size is None:
                continue
            out.append(
                ModelQuant(repo=repo, filename=name, quant=parse_quant(name), size_bytes=sib.size)
            )
        return out


@dataclass(frozen=True)
class RunnableQuant:
    quant: ModelQuant
    installed: bool


def usable_budget_gb(memory_budget_gb: float | None, *, unified: bool) -> float | None:
    """Memory available for a model after OS/runtime headroom, or None if the
    accelerator budget is unknown."""
    if memory_budget_gb is None:
        return None
    headroom = _UNIFIED_HEADROOM_GB if unified else _DISCRETE_HEADROOM_GB
    return max(0.0, memory_budget_gb - headroom)


def quant_fits(quant: ModelQuant, usable_gb: float | None, disk_free_bytes: int) -> bool:
    """A quant fits if its load estimate is within the usable accelerator budget
    AND its file fits free disk. Unknown budget → memory not gating (disk only)."""
    if quant.size_bytes > disk_free_bytes:
        return False
    if usable_gb is None:
        return True
    return quant.size_gb * _LOAD_OVERHEAD <= usable_gb


def runnable_models(
    browser: HfBrowser,
    *,
    memory_budget_gb: float | None,
    unified: bool,
    disk_free_bytes: int,
    installed_ids: frozenset[str] = frozenset(),
    limit: int = 30,
    per_model: int = 1,
) -> list[RunnableQuant]:
    """Query HF and return the quants this host can run — the largest-fitting
    quant per model (best quality that fits), popular models first."""
    usable = usable_budget_gb(memory_budget_gb, unified=unified)
    out: list[RunnableQuant] = []
    for repo in browser.search(limit=limit):
        fitting = [q for q in browser.quants(repo) if quant_fits(q, usable, disk_free_bytes)]
        # Largest-fitting first = best quality that runs.
        fitting.sort(key=lambda q: q.size_bytes, reverse=True)
        for q in fitting[:per_model]:
            out.append(RunnableQuant(quant=q, installed=q.model_id in installed_ids))
    return out
