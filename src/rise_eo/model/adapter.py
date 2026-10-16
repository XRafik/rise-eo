"""The ModelAdapter contract.

Spec: specs/18-model-adapter.md

A project team plugs a model into fsd by writing a small adapter class. fsd **duck-types**
it — no forced base class (the OQ-3 precedent: a documented signature, not an ABC, until a
real 2nd implementation exists). We ship:

- `ModelAdapter` — a `typing.Protocol` for documentation + static typing (structural, not
  inherited).
- `BaseModelAdapter` — an *optional* convenience with sensible defaults for `features`,
  `datacube_to_X`, and `to_output`, so a typical adapter is ~10 lines (declarations + `load`
  + `predict`).
- `Output` — what `to_output` returns: a `(bands, H, W)` array + dtype/nodata/band-names.

The feature transform is declared **once** (`feature_sequence`, a `rise_eo.bands.modify` sequence,
or the `features()` escape hatch) and run by fsd in BOTH training and inference — the F1
anti-skew invariant. See `specs/18-model-adapter.md` and the bundle explainer.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

__all__ = ["Output", "ModelAdapter", "BaseModelAdapter"]


class Output:
    """Model output for one datacube: `array` is `(bands, H, W)`.

    A tiny value object (not a namedtuple, so `isinstance` checks are unambiguous).
    """

    __slots__ = ("array", "dtype", "nodata", "band_names")

    def __init__(self, array: np.ndarray, dtype: str, nodata, band_names: list[str]):
        self.array = array
        self.dtype = dtype
        self.nodata = nodata
        self.band_names = band_names

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (f"Output(array.shape={self.array.shape}, dtype={self.dtype!r}, "
                f"nodata={self.nodata!r}, band_names={self.band_names!r})")


@runtime_checkable
class ModelAdapter(Protocol):
    """Structural contract fsd expects. Implement it (any framework); fsd duck-types.

    Declarations (read at **preflight**, before any heavy work):
        required_bands   : bands the datacube must contain.
        n_timestamps     : T the model was trained on; fsd asserts it equals the derived T.
        output_dtype     : e.g. "uint8".
        output_nodata    : e.g. 255.
        output_band_names: 1 name => categorical map; N => probabilities/regression.

    Feature engineering (F1 — ONE definition, run by fsd at train AND inference):
        feature_sequence : a `rise_eo.bands.modify` sequence (primary), or None.
        features(...)    : escape hatch when the sequence can't express it.

    Lifecycle (you own; any framework):
        load, datacube_to_X, predict, to_output.
    """

    required_bands: list[str]
    n_timestamps: int
    output_dtype: str
    output_nodata: Any
    output_band_names: list[str]
    feature_sequence: list[tuple[Callable, dict]] | None

    def load(self) -> None: ...
    def datacube_to_X(self, feats: np.ndarray, band_indices: dict) -> np.ndarray: ...
    def predict(self, X_chunk: np.ndarray) -> np.ndarray: ...
    def to_output(self, raw: np.ndarray, hw: tuple[int, int]) -> Output: ...


class BaseModelAdapter:
    """Optional base with defaults for `features`, `datacube_to_X`, `to_output`.

    Subclass and declare the spec attributes + implement `load` and `predict`; the rest is
    usually inherited. `artifacts` is injected by `rise_eo.model.bundle.load` with **absolute**
    paths (`{name: path}`) before `load()` is called.
    """

    # Subclasses override these declarations.
    required_bands: list[str] = []
    n_timestamps: int = 0
    output_dtype: str = "uint8"
    output_nodata: Any = 255
    output_band_names: list[str] = ["output"]
    feature_sequence: list[tuple[Callable, dict]] | None = None

    #: {name: absolute path}; set by bundle.load(), else empty for live use.
    artifacts: dict[str, str] = {}

    def load(self) -> None:
        """Load the artifact(s) into memory (once per worker). Override me."""

    def features(self, data5d: np.ndarray, band_indices: dict):
        """Escape hatch for feature engineering the sequence can't express.

        Only called when `feature_sequence is None`. Default raises to make the mistake
        loud (declare a `feature_sequence` or override this).
        """
        raise NotImplementedError(
            "declare a `feature_sequence` or override `features()` on your adapter."
        )

    def datacube_to_X(self, feats: np.ndarray, band_indices: dict) -> np.ndarray:
        """`(T, H, W, B)` features -> `(H*W, T*B)` model input (the legacy reshape)."""
        nt, nh, nw, nb = feats.shape
        return feats.reshape(nt, nh * nw, nb).swapaxes(0, 1).reshape(nh * nw, nt * nb)

    def predict(self, X_chunk: np.ndarray) -> np.ndarray:
        raise NotImplementedError("implement predict() on your adapter.")

    def to_output(self, raw: np.ndarray, hw: tuple[int, int]) -> Output:
        """Scattered per-pixel predictions -> `(bands, H, W)` Output.

        `raw` is `(H*W,)` (categorical) or `(H*W, k)` (probabilities/regression), already
        NaN-masked + nodata-filled by fsd. Default reshapes; override for custom encodings.
        """
        h, w = hw
        raw = np.asarray(raw)
        if raw.ndim == 1:
            arr = raw.reshape(1, h, w)
        else:
            arr = raw.reshape(h, w, -1).transpose(2, 0, 1)  # (k, H, W)
        return Output(
            array=arr.astype(self.output_dtype),
            dtype=self.output_dtype,
            nodata=self.output_nodata,
            band_names=self.output_band_names,
        )
