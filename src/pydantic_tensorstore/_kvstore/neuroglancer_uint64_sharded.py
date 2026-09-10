"""Neuroglancer uint64 sharded kvstore adapter and its sharding spec."""

from __future__ import annotations

from typing import Annotated, Literal

from annotated_types import Interval
from pydantic import Field

from pydantic_tensorstore._core.base import TensorStoreModel
from pydantic_tensorstore._kvstore.base import _CachedKvStoreAdapter


class NeuroglancerShardingSpec(TensorStoreModel):
    """Neuroglancer `neuroglancer_uint64_sharded_v1` sharding metadata."""

    type_: Literal["neuroglancer_uint64_sharded_v1"] = Field(
        default="neuroglancer_uint64_sharded_v1", alias="@type"
    )
    preshift_bits: Annotated[int, Interval(ge=0, le=64)] = Field(
        description="Number of low-order bits of the chunk ID that do not contribute "
        "to the hashed chunk ID.",
    )
    hash: Literal["identity", "murmurhash3_x86_128"] = Field(
        description="Hash function for sharding"
    )
    minishard_bits: Annotated[int, Interval(ge=0, le=64)] = Field(
        description="Number of bits of the hashed chunk ID that "
        "determine the minishard number."
    )
    shard_bits: Annotated[int, Interval(ge=0, le=64)] = Field(
        description="Number of bits of the hashed chunk ID that "
        "determine the shard number."
    )
    minishard_index_encoding: Literal["gzip", "raw"] | None = Field(
        default=None,
        description='Encoding of the minishard index. Default: `"raw"`.',
    )
    data_encoding: Literal["gzip", "raw"] | None = Field(
        default=None,
        description='Encoding of the chunk data. Default: `"raw"`.',
    )


class NeuroglancerUint64ShardedKvStore(_CachedKvStoreAdapter):
    """Read/write adapter for the Neuroglancer Precomputed sharded format."""

    driver: Literal["neuroglancer_uint64_sharded"] = "neuroglancer_uint64_sharded"
    metadata: NeuroglancerShardingSpec = Field(description="Sharding format.")
