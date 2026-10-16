"""Demo adapter for the end-to-end run (spec 19).

Band-limited on purpose: `satellite_benchmark/` only has B04/B08/B8A, so this uses NDVI + SAVI
(both from B04/B08) rather than the 9-band `examples/eurocrops_rf.py`. Importable as
`adapters:DemoRF` so it can back a model bundle (run the demo from the `demos/` dir, or with
`demos/` on `PYTHONPATH`).
"""

from __future__ import annotations

from rise_eo.bands import modify
from rise_eo.model import BaseModelAdapter


class DemoRF(BaseModelAdapter):
    required_bands = ["B04", "B08"]
    n_timestamps = 0                        # model-determined: the run sets it per instance and the
                                            # bundle records it (not hardcoded -> bundle-safe for any T)
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["crop_class"]
    feature_sequence = [
        (modify.mask_invalid_and_interpolate, {}),
        (modify.compute_bands, dict(bands_to_compute=["NDVI", "SAVI"])),
        (modify.remove_bands, dict(bands_to_remove=["B04", "B08", "B8A"])),  # -> NDVI, SAVI
    ]

    def load(self):
        import joblib
        self.clf, self.le = joblib.load(self.artifacts["model"])

    def predict(self, X_chunk):
        return self.clf.predict(X_chunk).astype("uint8")
