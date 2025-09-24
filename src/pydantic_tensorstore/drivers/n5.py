"""N5 driver specification for N5 format."""

from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from annotated_types import Interval, Le
from pydantic import BaseModel, Field, field_validator

from pydantic_tensorstore.core.codec import CodecBase
from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec


class N5Metadata(BaseModel):
    """N5 metadata specification.

    N5 is a chunked array storage format similar to Zarr,
    developed by the Saalfeld lab at HHMI Janelia.
    """

    model_config: ClassVar = {"extra": "allow"}

    dimensions: list[int] = Field(
        description="Array dimensions",
    )

    blockSize: list[int] = Field(
        description="Block (chunk) size for each dimension",
    )

    dataType: str = Field(
        description="N5 data type specification",
    )

    compression: dict[str, Any] = Field(
        default_factory=lambda: {"type": "raw"},
        description="Compression configuration",
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

    offset: list[float] | None = Field(
        default=None,
        description="Offset for each dimension",
    )

    @field_validator("dimensions", "blockSize", mode="before")
    @classmethod
    def validate_int_list(cls, v: Any) -> list[int]:
        """Validate integer list fields."""
        if not isinstance(v, list):
            raise ValueError("Must be a list of integers")

        for i, val in enumerate(v):
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"Element {i} must be a positive integer")

        return v

    @field_validator("dataType", mode="before")
    @classmethod
    def validate_data_type(cls, v: Any) -> str:
        """Validate N5 data type specification."""
        if not isinstance(v, str):
            raise ValueError("dataType must be a string")

        # N5 uses specific type names
        valid_types = {
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        }

        if v not in valid_types:
            raise ValueError(f"Invalid N5 dataType '{v}'. Valid types: {valid_types}")

        return v


class N5Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """N5 driver specification for N5 format."""

    driver: Literal["n5"] = "n5"

    metadata: N5Metadata | None = Field(
        default=None,
        description="N5 metadata specification",
    )


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
    blocksize: Annotated[int, Interval(ge=1, le=9)] = Field(
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


class N5Codec(CodecBase):
    """N5 codec specification."""

    driver: Literal["n5"] = "n5"
    compression: N5Compression | None = Field(
        default=None, description="N5 compression configuration"
    )
