# Simple Union type for all driver specs
from typing import Annotated, TypeAlias

from pydantic import Field

from pydantic_tensorstore.drivers.array import ArraySpec
from pydantic_tensorstore.drivers.n5 import N5Spec
from pydantic_tensorstore.drivers.neuroglancer_precomputed import (
    NeuroglancerPrecomputedSpec,
)
from pydantic_tensorstore.drivers.zarr import ZarrSpec
from pydantic_tensorstore.drivers.zarr3 import Zarr3Spec

TensorStoreSpec: TypeAlias = Annotated[
    ArraySpec | N5Spec | NeuroglancerPrecomputedSpec | ZarrSpec | Zarr3Spec,
    Field(discriminator="driver"),
]
