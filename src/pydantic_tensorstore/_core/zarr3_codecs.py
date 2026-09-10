"""Zarr v3 codec models (shared by the zarr3 driver and sharding kvstore)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from annotated_types import Ge, Interval
from pydantic import BeforeValidator, Field, PositiveInt

from pydantic_tensorstore._core.base import TensorStoreModel


class _Zarr3SingleCodec(TensorStoreModel):
    """Base class for single Zarr3 codec specifications."""

    # name: str
    # configuration: BaseModel


class Zarr3BloscConfig(TensorStoreModel):
    """Configuration for the Blosc codec."""

    cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] | None = Field(
        default=None, description='Compression algorithm. Default: `"lz4"`.'
    )
    clevel: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None, description="Compression level (0-9). Default: `5`."
    )
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] | None = Field(
        default=None,
        description="Shuffle filter. Default: `bitshuffle` for 1-byte elements, "
        "otherwise `shuffle`.",
    )
    typesize: Annotated[int, Interval(ge=1, le=255)] | None = Field(
        default=None, description="Stride in bytes for shuffling."
    )
    blocksize: Annotated[int, Ge(0)] | None = Field(
        default=None,
        description="Blosc block size in bytes. Default: `0` (automatic).",
    )


class Zarr3CodecBlosc(_Zarr3SingleCodec):
    """Specifies Blosc compression."""

    name: Literal["blosc"] = "blosc"
    configuration: Zarr3BloscConfig | None = None


class Zarr3BytesConfig(TensorStoreModel):
    """Configuration for the bytes codec."""

    endian: Literal["little", "big"] | None = Field(
        default=None, description="Byte order. Required for multi-byte data types."
    )


class Zarr3CodecBytes(_Zarr3SingleCodec):
    """Fixed-size encoding for numeric types."""

    name: Literal["bytes"] = "bytes"
    configuration: Zarr3BytesConfig | None = None


class Zarr3CRC32CConfig(TensorStoreModel):
    """No configuration options are supported."""


class Zarr3CodecCRC32C(_Zarr3SingleCodec):
    """Appends a CRC-32C checksum to detect data corruption."""

    name: Literal["crc32c"] = "crc32c"
    configuration: Zarr3CRC32CConfig | None = None


class Zarr3GzipConfig(TensorStoreModel):
    """Gzip codec configuration."""

    level: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None, description="Compression level (0-9). Default: `6`."
    )


class Zarr3CodecGzip(_Zarr3SingleCodec):
    """Specifies gzip compression."""

    name: Literal["gzip"] = "gzip"
    configuration: Zarr3GzipConfig | None = None


class Zarr3ShardingIndexedConfig(TensorStoreModel):
    """Configuration for the sharding_indexed codec."""

    chunk_shape: list[PositiveInt] | None = Field(
        default=None, description="Shape of each sub-chunk."
    )
    codecs: Zarr3CodecChain | None = Field(
        default=None,
        description="Codec chain used to encode/decode individual sub-chunks.",
    )
    index_codecs: Zarr3CodecChain | None = Field(
        default=None,
        description="Shard index codec chain, used to encode/decode the shard index.",
    )
    index_location: Literal["start", "end"] | None = Field(
        default=None, description='Location of the shard index. Default: `"end"`.'
    )


class Zarr3CodecShardingIndexed(_Zarr3SingleCodec):
    """Sharding codec that enables hierarchical chunking."""

    name: Literal["sharding_indexed"] = "sharding_indexed"
    configuration: Zarr3ShardingIndexedConfig | None = None


class Zarr3TransposeConfig(TensorStoreModel):
    """Configuration for the transpose codec."""

    order: list[int] | Literal["C", "F"] | None = Field(
        default=None,
        description="Permutation of the dimensions, or `C`/`F` order. "
        "https://google.github.io/tensorstore/driver/zarr3/index.html#json-driver/zarr3/Codec/transpose.configuration.order",
    )


class Zarr3CodecTranspose(_Zarr3SingleCodec):
    """Transposes the dimensions of an array."""

    name: Literal["transpose"] = "transpose"
    configuration: Zarr3TransposeConfig | None = None


class Zarr3ZstdConfig(TensorStoreModel):
    """Zstd codec configuration."""

    level: Annotated[int, Interval(ge=-131072, le=22)] | None = Field(
        default=None,
        description="Compression level (-131072 to 22). Default: `1`.",
    )
    checksum: bool | None = Field(
        default=None, description="Include a checksum. Default: `false`."
    )


class Zarr3CodecZstd(_Zarr3SingleCodec):
    """Specifies Zstd compression."""

    name: Literal["zstd"] = "zstd"
    configuration: Zarr3ZstdConfig | None = None


def _str_to_codec(v: Any) -> Any:
    """A plain string is equivalent to an object with the string as its name.

    For example, "crc32c" is equivalent to {"name": "crc32c"}.
    """
    return {"name": v} if isinstance(v, str) else v


Zarr3SingleCodec: TypeAlias = Annotated[
    Zarr3CodecBlosc
    | Zarr3CodecBytes
    | Zarr3CodecCRC32C
    | Zarr3CodecGzip
    | Zarr3CodecShardingIndexed
    | Zarr3CodecTranspose
    | Zarr3CodecZstd,
    Field(discriminator="name"),
    BeforeValidator(_str_to_codec),
]
Zarr3CodecChain: TypeAlias = list[Zarr3SingleCodec]

Zarr3ShardingIndexedConfig.model_rebuild()
