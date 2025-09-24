"""N5 driver specification for N5 format."""

from typing import Annotated, ClassVar, Literal, TypeAlias

from annotated_types import Interval, Le
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

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


def _str_to_compression(v: str | dict) -> dict:
    r"""A plain string is equivalent to an object with the string as its type.

    For example, \"gzip\" is equivalent to {\"type\": \"gzip\"}.
    """
    if isinstance(v, str):
        return {"type": v}
    return v


class _N5CompressionBlosc(BaseModel):
    type: Literal["blosc"] = "blosc"
    cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] = Field(
        default="lz4", description="Blosc compression algorithm"
    )
    clevel: Annotated[int, Interval(ge=0, le=9)] = Field(
        default=5, description="Specifies the Blosc compression level to use."
    )
    shuffle: Literal[0, 1, 2] = Field(
        default=1, description="Specifies the Blosc shuffle filter to use."
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
    preset: Annotated[int, Interval(ge=0, le=9)] = Field(
        default=6,
        description="XZ compression preset level (0-9). "
        "Higher values provide better compression.",
    )


class _N5CompressionZstd(BaseModel):
    type: Literal["zstd"] = "zstd"
    level: Annotated[int, Le(22)] = Field(
        default=0,
        description="Zstandard compression level (≤22). "
        "Higher values provide better compression.",
    )


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
    BeforeValidator(_str_to_compression),
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

    @model_validator(mode="after")
    def _validate_array_consistency(self) -> Self:
        """Validate consistency between array dimensions and related fields."""
        if self.dimensions is not None:
            dimensions_len = len(self.dimensions)

            # Validate blockSize length matches dimensions
            if self.blockSize is not None:
                if len(self.blockSize) != dimensions_len:
                    raise ValueError(
                        f"blockSize length ({len(self.blockSize)}) must match "
                        f"dimensions length ({dimensions_len})"
                    )

            # Validate axes length matches dimensions
            if self.axes is not None:
                if len(self.axes) != dimensions_len:
                    raise ValueError(
                        f"axes length ({len(self.axes)}) must match "
                        f"dimensions length ({dimensions_len})"
                    )

            # Validate units length matches dimensions
            if self.units is not None:
                if len(self.units) != dimensions_len:
                    raise ValueError(
                        f"units length ({len(self.units)}) must match "
                        f"dimensions length ({dimensions_len})"
                    )

            # Validate resolution length matches dimensions
            if self.resolution is not None:
                if len(self.resolution) != dimensions_len:
                    raise ValueError(
                        f"resolution length ({len(self.resolution)}) must match "
                        f"dimensions length ({dimensions_len})"
                    )

        return self


# Rebuild model to resolve forward references
N5Metadata.model_rebuild()


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
