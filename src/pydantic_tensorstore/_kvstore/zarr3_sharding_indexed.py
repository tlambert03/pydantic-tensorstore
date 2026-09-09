"""Zarr v3 sharding_indexed kvstore adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt

from pydantic_tensorstore._core.zarr3_codecs import Zarr3CodecChain
from pydantic_tensorstore._kvstore.base import _CachedKvStoreAdapter


class Zarr3ShardingIndexedKvStore(_CachedKvStoreAdapter):
    """Read/write adapter for the zarr v3 `sharding_indexed` format."""

    driver: Literal["zarr3_sharding_indexed"] = "zarr3_sharding_indexed"
    grid_shape: list[PositiveInt] = Field(
        description="Shape of the grid of entries in the shard."
    )
    index_codecs: Zarr3CodecChain = Field(
        description="Codec chain for encoding/decoding the shard index."
    )
    index_location: Literal["start", "end"] | None = Field(
        default=None, description='Location of the shard index. Default: `"end"`.'
    )
