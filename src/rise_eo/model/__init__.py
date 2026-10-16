"""rise-eo model contract: plug a model into rise-eo via a small ModelAdapter.

Spec: specs/18-model-adapter.md

Public surface:
    ModelAdapter, BaseModelAdapter, Output  — the contract (adapter.py).
    apply_features, median_per_id            — the F1 chokepoint + F4 reducer (features.py).
    infer_datacube, run_local                — the inference engine (engine.py).
    bundle                                   — save/load the self-describing model bundle.
    verify_image                             — does an inference image run this bundle?
    registry                                 — a resolvable name for a published bundle
"""

from rise_eo.model import bundle, registry
from rise_eo.model.adapter import BaseModelAdapter, ModelAdapter, Output
from rise_eo.model.engine import infer_datacube, infer_datacube_to_cog, run_local
from rise_eo.model.features import apply_features, median_per_id, resolve_aggregate
from rise_eo.model.verify_image import verify_image

load_bundle = bundle.load
save_bundle = bundle.save

__all__ = [
    "ModelAdapter",
    "BaseModelAdapter",
    "Output",
    "apply_features",
    "median_per_id",
    "resolve_aggregate",
    "infer_datacube",
    "infer_datacube_to_cog",
    "run_local",
    "bundle",
    "load_bundle",
    "save_bundle",
    "verify_image",
    "registry",
]
