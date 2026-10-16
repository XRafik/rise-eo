from rise_eo.bands import modify
from rise_eo.model.adapter import BaseModelAdapter

class CropRF(BaseModelAdapter):
    required_bands = ["B04", "B08"]        # raw bands SEQ consumes
    n_timestamps = 0                       # set per-instance at bundle time
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["crop_class"]
    feature_sequence = [
        (modify.mask_invalid_and_interpolate, {}),
        (modify.compute_bands, dict(bands_to_compute=["NDVI", "SAVI"])),
        (modify.remove_bands, dict(bands_to_remove=["B04", "B08", "B8A"])),
    ]

    def load(self):
        import joblib
        self.clf, self.le = joblib.load(self.artifacts["model"])

    def predict(self, X_chunk):
        return self.clf.predict(X_chunk).astype("uint8")

class CropRFS1(CropRF):
    """The same RF over Sentinel-1 vv/vh backscatter instead of optical indices."""
    required_bands = ["vv", "vh"]
    feature_sequence = [
        (modify.mask_invalid_and_interpolate, {}),
    ]
