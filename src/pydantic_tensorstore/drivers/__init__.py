"""TensorStore driver specifications."""

from typing import Annotated, TypeAlias

from pydantic import Field

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


TensorStoreSpec: TypeAlias = Annotated[
    ArraySpec | N5Spec | NeuroglancerPrecomputedSpec | ZarrSpec | Zarr3Spec,
    Field(discriminator="driver"),
]
