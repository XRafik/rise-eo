"""Example ModelAdapter: the EuroCrops Random-Forest crop classifier (spec 18).

Reproduces the legacy `demo_02_model_train` + `demo_03_model_deploy` through the fsd contract.
This file is an **example**, not part of the fsd wheel — to use it in a bundle, make it
importable (`pip install -e .` your project, or put this dir on `PYTHONPATH`) so the ref
`eurocrops_rf:EuroCropsRF` resolves.

Train side (your code, sklearn — fsd does not train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import joblib
    td = rise_eo.create_training_data(..., adapter=EuroCropsRF())   # writes features.npy
    d = td.load()
    y = LabelEncoder().fit(d["feature_labels"])
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1).fit(
        d["features"].reshape(len(d["features"]), -1), y.transform(d["feature_labels"]))
    joblib.dump((clf, y), "rf.joblib")

Package + deploy:
    from rise_eo.model import bundle
    bundle.save(EuroCropsRF(), {"model": "rf.joblib"}, "eurocrops_bundle")
    rise_eo.run_inference("eurocrops_bundle", inference_datacubes="…/grids", output_folderpath="…/out")
"""

from __future__ import annotations

import joblib
import numpy as np

from rise_eo.bands import modify
from rise_eo.model import BaseModelAdapter

S2_L2A_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]


class EuroCropsRF(BaseModelAdapter):
    # --- spec (read at preflight, before any heavy work) ---
    required_bands = S2_L2A_BANDS
    n_timestamps = 19                    # full-year, 20-day calendar mosaic (spec 15)
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["crop_class"]   # one band => categorical map (label-encoded ints)

    # --- F1: ONE feature transform, run by fsd at BOTH train and inference ---
    feature_sequence = [
        (modify.mask_invalid_and_interpolate, {}),
        (modify.compute_bands, dict(bands_to_compute=["NDVI", "NDRE", "GCVI", "SAVI"])),
        (modify.remove_bands, dict(bands_to_remove=S2_L2A_BANDS)),
    ]

    # --- lifecycle (you own; sklearn here, any framework in general) ---
    def load(self) -> None:
        # self.artifacts["model"] is an absolute path injected by rise_eo.model.bundle.load.
        self.clf, self.label_encoder = joblib.load(self.artifacts["model"])

    def predict(self, X_chunk: np.ndarray) -> np.ndarray:
        # fsd hands us valid (non-NaN) rows; return label-encoded class ids.
        return self.clf.predict(X_chunk).astype("uint8")

    # datacube_to_X and to_output are inherited from BaseModelAdapter (the legacy reshape +
    # single-band categorical packaging); override only if your model needs something else.
