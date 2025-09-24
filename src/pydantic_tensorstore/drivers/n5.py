"""N5 driver specification for N5 format."""

from typing import Annotated, ClassVar, Literal, TypeAlias

from annotated_types import Interval, Le
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
)

from pydantic_tensorstore._types import DataType
from pydantic_tensorstore.core.codec import CodecBase
from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec

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


class _N5CompressionBlosc(BaseModel):
    type: Literal["blosc"] = "blosc"
    cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] = Field(
        description="Blosc compression algorithm"
    )
    clevel: Annotated[int, Interval(ge=0, le=9)] = Field(
        description="Specifies the Blosc compression level to use."
    )
    shuffle: Literal[0, 1, 2] = Field(
        description="Specifies the Blosc shuffle filter to use."
    )


class _N5CompressionBzip2(BaseModel):
    type: Literal["bzip2"] = "bzip2"
    blockSize: Annotated[int, Interval(ge=1, le=9)] = Field(
        default=9,
        description="Specifies the bzip2 block size to use (in units of 100KB), "
        "which also determine the compression level.",
    )


class _N5CompressionGzip(BaseModel):
    type: Literal["gzip"] = "gzip"
    level: Annotated[int, Interval(ge=-1, le=9)] = Field(
        default=-1,
        description=(
            "Specifies the gzip compression level to use."
            "Level 0 indicates no compression (fastest), while level 9 indicates "
            "the best compression ratio (slowest). The default value of -1 indicates "
            "to use the zlib default compression level (equal to 6)."
        ),
    )
    useZlib: bool = Field(
        default=False,
        description="If true, use zlib instead of gzip.",
    )


class _N5CompressionRaw(BaseModel):
    """Chunks are encoded directly as big endian values without compression."""

    type: Literal["raw"] = "raw"


class _N5CompressionXZ(BaseModel):
    type: Literal["xz"] = "xz"
    preset: Annotated[int, Interval(ge=0, le=9)] = 6


class _N5CompressionZstd(BaseModel):
    type: Literal["zstd"] = "zstd"
    level: Annotated[int, Le(22)] = 0


N5Compression: TypeAlias = Annotated[
    (
        _N5CompressionBlosc
        | _N5CompressionBzip2
        | _N5CompressionGzip
        | _N5CompressionRaw
        | _N5CompressionXZ
        | _N5CompressionZstd
    ),
    Field(discriminator="type"),
]


class N5Metadata(BaseModel):
    """N5 metadata specification.

    N5 is a chunked array storage format similar to Zarr,
    developed by the Saalfeld lab at HHMI Janelia.
    """

    model_config: ClassVar = {"extra": "allow"}

    dimensions: list[NonNegativeInt] | None = Field(
        default=None,
        description="Array dimensions",
    )

    blockSize: list[PositiveInt] | None = Field(
        default=None,
        description="Block (chunk) size for each dimension",
    )

    dataType: N5DataType | None = Field(
        default=None,
        description="N5 data type specification",
    )

    axes: list[str] | None = Field(
        default=None,
        description="Axis labels for each dimension",
    )

    units: list[str] | None = Field(
        default=None,
        description="Physical units for each dimension",
    )

    resolution: list[float] | None = Field(
        default=None,
        description="Resolution (pixel size) for each dimension",
    )

    compression: N5Compression | None = Field(
        default=None,
        description="Compression configuration",
    )


class N5Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """N5 driver specification for N5 format."""

    driver: Literal["n5"] = "n5"

    metadata: N5Metadata | None = Field(
        default=None,
        description="N5 metadata specification",
    )


class N5Codec(CodecBase):
    """N5 codec specification."""

    driver: Literal["n5"] = "n5"
    compression: N5Compression | None = Field(
        default=None, description="N5 compression configuration"
    )
