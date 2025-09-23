"""TensorStore driver specifications."""

import sys
from typing import Annotated, Any, TypeAlias

from pydantic import BeforeValidator, Field

from .array import ArraySpec
from .n5 import N5Spec
from .neuroglancer_precomputed import NeuroglancerPrecomputedSpec
from .zarr import ZarrSpec
from .zarr3 import Zarr3Spec

__all__ = [
    "ArraySpec",
    "N5Spec",
    "NeuroglancerPrecomputedSpec",
    "Zarr3Spec",
    "ZarrSpec",
]


def _cast_to_spec_dict(obj: Any) -> Any:
    if ts := sys.modules.get("tensorstore"):
        if isinstance(obj, ts.TensorStore):
            obj = obj.spec()
        if isinstance(obj, ts.Spec):
            return obj.to_json()
    return obj


TensorStoreSpec: TypeAlias = Annotated[
    ArraySpec | N5Spec | NeuroglancerPrecomputedSpec | ZarrSpec | Zarr3Spec,
    Field(discriminator="driver"),
    BeforeValidator(_cast_to_spec_dict),
]
