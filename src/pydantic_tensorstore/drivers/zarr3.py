"""Zarr3 driver specification for Zarr v3 format."""

from typing import Annotated, Any, Literal, TypeAlias

from annotated_types import Ge, Interval
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    NonNegativeInt,
    PositiveInt,
)

from pydantic_tensorstore._types import DataType
from pydantic_tensorstore.core.codec import CodecBase
from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec

VALID_ZARR3_DTYPES: set[DataType] = {
    DataType.BFLOAT16,
    DataType.BOOL,
    DataType.COMPLEX128,
    DataType.COMPLEX64,
    DataType.FLOAT16,
    DataType.FLOAT32,
    DataType.FLOAT64,
    DataType.INT4,
    DataType.INT8,
    DataType.INT16,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT8,
    DataType.UINT16,
    DataType.UINT32,
    DataType.UINT64,
}

Zarr3DataType: TypeAlias = Annotated[
    DataType,
    AfterValidator(
        lambda v: v in VALID_ZARR3_DTYPES
        or ValueError(
            f"Invalid Zarr3 data type: {v}. Must be one of {VALID_ZARR3_DTYPES}"
        )
    ),
]


class _Zarr3SingleCodec(BaseModel):
    """Base class for single Zarr3 codec specifications."""

    # name: str
    # configuration: BaseModel


class Zarr3CodecBlosc(_Zarr3SingleCodec):
    """Specifies Blosc compression."""

    class BloscConfig(BaseModel):
        """Configuration for the Blosc codec."""

        cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] = Field(
            default="lz4",
            description="Compression algorithm",
        )
        clevel: Annotated[int, Interval(ge=0, le=5)] = Field(
            default=5, ge=0, le=9, description="Compression level (0-9)"
        )
        shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] | None = Field(
            default=None, description="Shuffle filter"
        )
        typesize: Annotated[int, Interval(ge=1, le=255)] | None = Field(
            default=1, description="Specifies the stride in bytes for shuffling."
        )
        blocksize: Annotated[int, Ge(0)] | None = Field(
            default=0,
            description="Blosc block size in bytes. The default value of 0 causes the "
            "block size to be chosen automatically.",
        )

    name: Literal["blosc"] = "blosc"
    configuration: BloscConfig | None = Field(
        default=None, description="Blosc codec configuration"
    )


class Zarr3CodecBytes(_Zarr3SingleCodec):
    """Fixed-size encoding for numeric types."""

    class BytesConfig(BaseModel):
        """Configuration for the bytes codec."""

        endial: Literal["little", "big"] | None = Field(
            default=None, description="Byte order"
        )

    name: Literal["bytes"] = "bytes"
    configuration: BytesConfig | None = Field(
        default=None, description="Bytes codec configuration"
    )


class Zarr3CodecBCrc32c(_Zarr3SingleCodec):
    """Appends a CRC-32C checksum to detect data corruption."""

    class Crc32cConfig(BaseModel):
        """No configuration options are supported."""

    name: Literal["crc32c"] = "crc32c"
    configuration: Crc32cConfig | None = Field(
        default=None, description="No configuration options are supported."
    )


class Zarr3CodecGzip(_Zarr3SingleCodec):
    """Specifies gzip compression."""

    class GzipConfig(BaseModel):
        """Gzip codec configuration. (not documented in tensorstore)."""

        level: Annotated[int, Interval(ge=-1, le=9)] = Field(
            default=-1, description="Compression level (0-9)"
        )

    name: Literal["gzip"] = "gzip"
    configuration: GzipConfig | None = Field(
        default=None, description="Gzip codec configuration"
    )


class Zarr3CodecShardingIndexed(_Zarr3SingleCodec):
    """Sharding codec that enables hierarchical chunking."""

    class ShardingIndexedConfig(BaseModel):
        """Configuration for the sharding indexed codec."""

        chunk_shape: list[PositiveInt] | None = Field(
            default=None, description="Shape of each sub-chunk."
        )
        codecs: "Zarr3CodecChain | None" = Field(
            default=None,
            description=(
                "Sub-chunk codec chain, used to encode/decode individual sub-chunks."
            ),
        )
        index_codecs: "Zarr3CodecChain | None" = Field(
            default=None,
            description=(
                "Shard index codec chain, used to encode/decode the shard index."
            ),
        )
        index_location: Literal["start", "end"] = Field(
            default="end", description="Location of the shard index."
        )

    name: Literal["sharding_indexed"] = "sharding_indexed"
    configuration: ShardingIndexedConfig | None = Field(
        default=None, description="Sharding indexed codec configuration"
    )


class Zarr3CodecTranspose(_Zarr3SingleCodec):
    """Transposes the dimensions of an array."""

    class TransposeConfig(BaseModel):
        """Configuration for the transpose codec."""

        order: list[int | Literal["F", "C"]] | None = Field(
            default=None,
            description=(
                "Permutation of the dimensions. "
                "https://google.github.io/tensorstore/driver/zarr3/index.html#json-driver/zarr3/Codec/transpose.configuration.order"
            ),
        )

    name: Literal["transpose"] = "transpose"
    configuration: TransposeConfig | None = Field(
        default=None, description="Transpose codec configuration"
    )


class Zarr3CodecZstd(_Zarr3SingleCodec):
    """Specifies Zstd compression."""

    class ZstdConfig(BaseModel):
        """Zstd codec configuration."""

        level: Annotated[int, Interval(ge=-131072, le=22)] = Field(
            default=1,
            description="Compression level (-131072 to 22). Higher level provides "
            "improved density at the cost of compression speed.",
        )
        checksum: bool = Field(
            default=False, description="Whether to include a checksum."
        )

    name: Literal["zstd"] = "zstd"
    configuration: ZstdConfig | None = Field(
        default=None, description="Zstandard codec configuration"
    )


def _str_to_codec(v: str) -> dict[str, Any]:
    """A plain string is equivalent to an object with the string as its name.

    For example, "crc32c" is equivalent to {"name": "crc32c"}.
    """
    if isinstance(v, str):
        return {"name": v}
    return v


#  TODO
Zarr3SingleCodec: TypeAlias = Annotated[
    Zarr3CodecBlosc
    | Zarr3CodecBytes
    | Zarr3CodecBCrc32c
    | Zarr3CodecGzip
    | Zarr3CodecShardingIndexed
    | Zarr3CodecTranspose
    | Zarr3CodecZstd,
    BeforeValidator(_str_to_codec),
]
Zarr3CodecChain: TypeAlias = list[Zarr3SingleCodec]
Zarr3CodecShardingIndexed.ShardingIndexedConfig.model_rebuild()


class _ZarrChunkConfiguration(BaseModel):
    chunk_shape: list[PositiveInt] | None = Field(
        default=None,
        description="""Chunk dimensions.

    Specifies the chunk size for each dimension. Must have the same length as shape. If
    not specified when creating a new array, the chunk dimensions are chosen
    automatically according to the Schema.chunk_layout. If specified when creating a new
    array, the chunk dimensions must be compatible with the Schema.chunk_layout. When
    opening an existing array, the specified chunk dimensions must match the existing
    chunk dimensions.
    """,
    )


class _ZarrChunkGrid(BaseModel):
    name: Literal["regular"] = Field(
        default="regular",
        description="Chunk grid type (only 'regular' is supported)",
    )
    configuration: _ZarrChunkConfiguration


class _DefaultChunkKeyEncoding(BaseModel):
    """Default chunk key encoding."""

    name: Literal["default"] = "default"
    configuration: dict[str, Any] = Field(
        default_factory=lambda: {"separator": "/"},
        description="Default chunk key encoding configuration",
    )


class _V2ChunkKeyEncoding(BaseModel):
    """Default chunk key encoding."""

    name: Literal["v2"] = "v2"
    configuration: dict[str, Any] = Field(
        default_factory=lambda: {"separator": "."},
        description="Default chunk key encoding configuration",
    )


Zarr3ChunkKeyEncoding = _DefaultChunkKeyEncoding | _V2ChunkKeyEncoding


class Zarr3Metadata(BaseModel):
    """Zarr v3 metadata specification.

    Zarr v3 introduces new features like sharding, variable chunks,
    and improved codec pipelines.
    """

    zarr_format: Literal[3] = 3
    node_type: Literal["array"] = "array"

    shape: list[NonNegativeInt] | None = Field(
        default=None,
        description=(
            "Array shape. Required when creating a new array "
            "if the `Schema.domain` is not otherwise specified."
        ),
    )

    data_type: Zarr3DataType | None = Field(
        default=None,
        description="Data type specification",
    )

    chunk_grid: _ZarrChunkGrid | None = Field(
        default=None,
        description="Chunk grid specification",
    )

    chunk_key_encoding: Zarr3ChunkKeyEncoding | None = Field(
        default=None,
        description="Chunk key encoding configuration",
    )

    fill_value: Any = Field(
        default=None,
        description="Fill value for uninitialized chunks",
    )

    codecs: Zarr3CodecChain | None = Field(
        default=None,
        description="Codec pipeline for compression and encoding",
    )

    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined attributes",
    )

    dimension_names: list[str | None] | None = Field(
        default=None,
        description="Names for each dimension",
    )


class Zarr3Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Zarr3 driver specification for Zarr v3 format."""

    driver: Literal["zarr3"] = "zarr3"

    metadata: Zarr3Metadata | None = Field(
        default=None,
        description="Zarr v3 metadata specification",
    )


class Zarr3Codec(CodecBase):
    """Zarr3 codec specification."""

    driver: Literal["zarr3"] = "zarr3"
    codecs: Zarr3CodecChain | None = Field(
        default=None, description="Specifies a chain of codecs."
    )
