"""fsd — download satellite imagery (CDSE, MPC) and build datacubes. See specs/ for the design.

High-level API (specs/16): `fsd.download`, `fsd.create_training_data`,
`fsd.flatten_training_data`, `fsd.run_inference`, `fsd.verify_adapter` and `fsd.deploy`.
These are the verbs users call; the modules under fsd.* are the internals.
"""

import importlib.metadata

from fsd import config
from fsd.api import (
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
from fsd.model import BaseModelAdapter, ModelAdapter, Output, load_bundle, save_bundle

__version__ = importlib.metadata.version("fsd")

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
