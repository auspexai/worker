"""Worker model management (W-M) — the BYOM onramp.

Helps a volunteer acquire models the network wants, into the local model store
the §9 #37 executor dispatch reads (`<data_dir>/models/<model_id>/`). The
platform never distributes weights (§5.8); this is the supply side.
"""

from __future__ import annotations

from auspexai_worker.models.catalog import (
    BundledCatalogSource,
    CatalogSource,
    FileCatalogSource,
    ModelCatalog,
    ModelCatalogEntry,
)
from auspexai_worker.models.recommend import WorkerResources, recommend, survey_resources
from auspexai_worker.models.store import ModelStore

__all__ = [
    "BundledCatalogSource",
    "CatalogSource",
    "FileCatalogSource",
    "ModelCatalog",
    "ModelCatalogEntry",
    "ModelStore",
    "WorkerResources",
    "recommend",
    "survey_resources",
]
