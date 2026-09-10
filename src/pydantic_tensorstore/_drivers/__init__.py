"""TensorStore driver specifications."""

from __future__ import annotations

import sys
from typing import Annotated, Any, TypeAlias

from pydantic import BeforeValidator, Field

from pydantic_tensorstore._core.schema import Schema
from pydantic_tensorstore._core.spec import TensorStoreAdapterSpec
from pydantic_tensorstore._kvstore import KvStore

from .array import ArraySpec
from .auto import AutoSpec
from .cast import CastSpec
from .downsample import DownsampleSpec
from .image import AvifSpec, BmpSpec, JpegSpec, PngSpec, TiffSpec, WebpSpec
from .json import JsonSpec
from .n5 import N5Codec, N5Spec
from .neuroglancer_precomputed import (
    NeuroglancerPrecomputedCodec,
    NeuroglancerPrecomputedSpec,
)
from .stack import StackSpec
from .zarr import Zarr2Codec, Zarr2Spec
from .zarr3 import Zarr3Codec, Zarr3Spec

__all__ = [
    "ArraySpec",
    "AutoSpec",
    "AvifSpec",
    "BmpSpec",
    "CastSpec",
    "Codec",
    "DownsampleSpec",
    "JpegSpec",
    "JsonSpec",
    "N5Spec",
    "NeuroglancerPrecomputedSpec",
    "PngSpec",
    "StackSpec",
    "TensorStoreSpec",
    "TiffSpec",
    "WebpSpec",
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
    ArraySpec
    | AutoSpec
    | AvifSpec
    | BmpSpec
    | CastSpec
    | DownsampleSpec
    | JpegSpec
    | JsonSpec
    | N5Spec
    | NeuroglancerPrecomputedSpec
    | PngSpec
    | StackSpec
    | TiffSpec
    | WebpSpec
    | Zarr2Spec
    | Zarr3Spec,
    Field(discriminator="driver"),
    BeforeValidator(_cast_to_spec_dict),
]

Codec: TypeAlias = Annotated[
    N5Codec | NeuroglancerPrecomputedCodec | Zarr2Codec | Zarr3Codec,
    Field(discriminator="driver"),
]

# Resolve forward references to the unions defined above.
_ns = {"Codec": Codec, "KvStore": KvStore, "TensorStoreSpec": TensorStoreSpec}
for _model in (Schema, TensorStoreAdapterSpec, CastSpec, DownsampleSpec, StackSpec):
    _model.model_rebuild(_types_namespace=_ns)
