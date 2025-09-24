"""TensorStore driver specifications."""

import sys
from typing import Annotated, Any, TypeAlias

from pydantic import BeforeValidator, Field

from .array import ArraySpec
from .n5 import N5Codec, N5Spec
from .neuroglancer_precomputed import (
    NeuroglancerPrecomputedCodec,
    NeuroglancerPrecomputedSpec,
)
from .tiff import TiffSpec
from .zarr import Zarr2Codec, Zarr2Spec
from .zarr3 import Zarr3Codec, Zarr3Spec

__all__ = [
    "ArraySpec",
    "N5Spec",
    "NeuroglancerPrecomputedSpec",
    "TiffSpec",
    "Zarr2Spec",
    "Zarr3Spec",
]


def _cast_to_spec_dict(obj: Any) -> Any:
    if ts := sys.modules.get("tensorstore"):
        if isinstance(obj, ts.TensorStore):
            obj = obj.spec()
        if isinstance(obj, ts.Spec):
            return obj.to_json()
    return obj


TensorStoreSpec: TypeAlias = Annotated[
    ArraySpec | N5Spec | NeuroglancerPrecomputedSpec | TiffSpec | Zarr2Spec | Zarr3Spec,
    Field(discriminator="driver"),
    BeforeValidator(_cast_to_spec_dict),
]

Codec: TypeAlias = Annotated[
    N5Codec | NeuroglancerPrecomputedCodec | Zarr2Codec | Zarr3Codec,
    Field(discriminator="driver"),
]
