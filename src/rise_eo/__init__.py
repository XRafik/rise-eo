"""rise-eo — download satellite imagery (CDSE, MPC) and build datacubes. See specs/ for the design.

High-level API (specs/16): `rise_eo.download`, `rise_eo.create_training_data`,
`rise_eo.flatten_training_data`, `rise_eo.run_inference`, `rise_eo.verify_adapter` and `rise_eo.deploy`.
These are the verbs users call; the modules under rise-eo.* are the internals.
"""

import importlib.metadata

from rise_eo import config
from rise_eo.api import (
    InferenceResult,
    PreflightError,
    TrainingData,
    archive_catalog_filepath,
    compute_n_timestamps,
    create_training_data,
    deploy,
    download,
    flatten_training_data,
    run_inference,
    verify_adapter,
)
from rise_eo.model import BaseModelAdapter, ModelAdapter, Output, load_bundle, save_bundle

__version__ = importlib.metadata.version("rise-eo")

__all__ = [
    "BaseModelAdapter",
    "InferenceResult",
    "ModelAdapter",
    "Output",
    "PreflightError",
    "TrainingData",
    "__version__",
    "archive_catalog_filepath",
    "compute_n_timestamps",
    "config",
    "create_training_data",
    "deploy",
    "download",
    "flatten_training_data",
    "load_bundle",
    "run_inference",
    "save_bundle",
    "verify_adapter",
]
