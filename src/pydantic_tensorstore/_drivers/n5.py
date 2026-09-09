"""N5 driver specification for N5 format."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias

from annotated_types import Interval, Le
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from pydantic_tensorstore._core.base import TensorStoreModel
from pydantic_tensorstore._core.codec import CodecBase
from pydantic_tensorstore._core.spec import ChunkedTensorStoreKvStoreAdapterSpec
from pydantic_tensorstore._types import DataType

VALID_N5_DTYPES: set[DataType] = {
    DataType.FLOAT32,
    DataType.FLOAT64,
    DataType.INT8,
    DataType.INT16,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT8,
    DataType.UINT16,
    DataType.UINT32,
    DataType.UINT64,
}


def _validate_N5_dtype(v: DataType) -> DataType:
    if v not in VALID_N5_DTYPES:
        raise ValueError(f"Invalid N5 data type: {v}. Must be one of {VALID_N5_DTYPES}")
    return v


N5DataType: TypeAlias = Annotated[DataType, AfterValidator(_validate_N5_dtype)]


def _str_to_compression(v: Any) -> Any:
    """Convenience: a plain string `"gzip"` becomes `{"type": "gzip"}`.

    Note that tensorstore itself only accepts the object form.
    """
    return {"type": v} if isinstance(v, str) else v


class N5CompressionBlosc(TensorStoreModel):
    """Blosc compression."""

    type: Literal["blosc"] = "blosc"
    cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] = Field(
        description="Blosc compression algorithm"
    )
    clevel: Annotated[int, Interval(ge=0, le=9)] = Field(
        description="Blosc compression level."
    )
    shuffle: Literal[0, 1, 2] = Field(
        description="Shuffle filter: 0 none, 1 byte-wise, 2 bit-wise."
    )


class N5CompressionBzip2(TensorStoreModel):
    """bzip2 compression."""

    type: Literal["bzip2"] = "bzip2"
    blockSize: Annotated[int, Interval(ge=1, le=9)] | None = Field(
        default=None,
        description="bzip2 block size in units of 100KB, which also determines the "
        "compression level. Default: `9`.",
    )


class N5CompressionGzip(TensorStoreModel):
    """gzip (or zlib) compression."""

    type: Literal["gzip"] = "gzip"
    level: Annotated[int, Interval(ge=-1, le=9)] | None = Field(
        default=None,
        description="Compression level; 0 is none, 9 is best, -1 is the zlib default "
        "(6). Default: `-1`.",
    )
    useZlib: bool | None = Field(
        default=None, description="Use zlib instead of gzip. Default: `false`."
    )


class N5CompressionRaw(TensorStoreModel):
    """Chunks are encoded directly as big endian values without compression."""

    type: Literal["raw"] = "raw"


class N5CompressionXZ(TensorStoreModel):
    """xz compression."""

    type: Literal["xz"] = "xz"
    preset: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None,
        description="XZ preset level (0-9); higher is better compression. "
        "Default: `6`.",
    )


class N5CompressionZstd(TensorStoreModel):
    """Zstandard compression."""

    type: Literal["zstd"] = "zstd"
    level: Annotated[int, Le(22)] | None = Field(
        default=None,
        description="Zstandard level (<= 22); higher is better compression. "
        "Default: `0`.",
    )


N5Compression: TypeAlias = Annotated[
    (
        N5CompressionBlosc
        | N5CompressionBzip2
        | N5CompressionGzip
        | N5CompressionRaw
        | N5CompressionXZ
        | N5CompressionZstd
    ),
    Field(discriminator="type"),
    BeforeValidator(_str_to_compression),
]


class N5Metadata(TensorStoreModel):
    """N5 `attributes.json` metadata; all members optional.

    Arbitrary additional members are stored as N5 attributes without validation.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    dimensions: list[NonNegativeInt] | None = Field(
        default=None, description="Array dimensions"
    )
    blockSize: list[PositiveInt] | None = Field(
        default=None, description="Block (chunk) size for each dimension"
    )
    dataType: N5DataType | None = Field(
        default=None, description="N5 data type specification"
    )
    axes: list[str] | None = Field(
        default=None, description="Axis labels for each dimension"
    )
    units: list[str] | None = Field(
        default=None, description="Physical units for each dimension"
    )
    resolution: list[float] | None = Field(
        default=None, description="Multiplier for the physical units, per dimension"
    )
    compression: N5Compression | None = Field(
        default=None, description="Chunk compression method"
    )

    @model_validator(mode="after")
    def _validate_array_consistency(self) -> Self:
        """Validate that per-dimension fields match `dimensions` in length."""
        if self.dimensions is None:
            return self
        n = len(self.dimensions)
        for name in ("blockSize", "axes", "units", "resolution"):
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(
                    f"{name} length ({len(value)}) must match dimensions length ({n})"
                )
        return self


class N5Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """N5 driver specification for N5 format."""

    driver: Literal["n5"] = "n5"
    metadata: N5Metadata | None = None


class N5Codec(CodecBase):
    """N5 codec specification."""

    driver: Literal["n5"] = "n5"
    compression: N5Compression | None = Field(
        default=None, description="N5 compression configuration"
    )
