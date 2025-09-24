"""Zarr driver specification for Zarr v2 format."""

from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from annotated_types import Interval
from pydantic import BaseModel, Field, NonNegativeInt, field_validator

from pydantic_tensorstore._types import JsonObject
from pydantic_tensorstore.core.codec import CodecBase
from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec


class ZarrMetadata(BaseModel):
    """Zarr metadata specification.

    Controls Zarr-specific format options like compression,
    chunk shapes, and array metadata.
    """

    model_config: ClassVar = {"extra": "allow"}

    chunks: list[int] | None = Field(
        default=None,
        description="Chunk shape for storage",
    )

    compressor: JsonObject | None = Field(
        default=None,
        description="Compression configuration",
    )

    filters: list[JsonObject] | None = Field(
        default=None,
        description="Filter pipeline configuration",
    )

    fill_value: int | float | str | bool | None = Field(
        default=None,
        description="Fill value for uninitialized chunks",
    )

    order: Literal["C", "F"] = Field(
        default="C",
        description="Memory layout order (C=row-major, F=column-major)",
    )

    zarr_format: Literal[2] = Field(
        default=2,
        description="Zarr format version",
    )

    dimension_separator: str = Field(
        default=".",
        description="Separator for dimension names in chunk keys",
    )

    @field_validator("chunks", mode="before")
    @classmethod
    def validate_chunks(cls, v: Any) -> list[int] | None:
        """Validate chunk specification."""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("chunks must be a list")

        for i, chunk_size in enumerate(v):
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError(
                    f"Chunk size at dimension {i} must be a positive integer"
                )

        return v


class Zarr2Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Zarr driver specification for Zarr v2 format."""

    driver: Literal["zarr"] = "zarr"

    metadata: ZarrMetadata | None = Field(
        default=None,
        description="Zarr metadata specification",
    )


class _Zarr2Compressor(BaseModel):
    """The id member identifies the compressor.

    The remaining members are specific to the compressor.
    """

    # id: str


class Zarr2CompressorBlosc(_Zarr2Compressor):
    """Blosc compressor specification."""

    id: Literal["blosc"] = "blosc"
    cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] | None = Field(
        default=None,
        description="Specifies the compression method used by Blosc.",
    )
    clevel: Annotated[int, Field(ge=0, le=9)] = Field(
        default=5,
        description="Specifies the Blosc compression level to use. Higher values "
        "indicate more compression at the cost of compression speed.",
    )
    shuffle: Literal[-1, 0, 1, 2] = Field(
        default=-1,
        description="Specifies the Blosc shuffle filter to use. A value of 0 "
        "indicates no shuffle, 1 indicates byte-wise shuffle, and 2 indicates "
        "bit-wise shuffle. -1 is auto: Bit-wise shuffle if the element size is 1 byte, "
        "otherwise byte-wise shuffle.",
    )
    blocksize: NonNegativeInt | None = Field(
        default=0,
        description="Specifies the desired block size in bytes. The default value of "
        "0 indicates that the compressor should automatically choose a block size.",
    )


class Zarr2CompressorBz2(_Zarr2Compressor):
    """Bz2 compressor specification."""

    id: Literal["bz2"] = "bz2"
    level: Annotated[int, Interval(ge=1, le=9)] = Field(
        default=1,
        description="Specifies the bzip2 buffer size/compression level to use. "
        "A level of 1 indicates the smallest buffer (fastest), while level 9 indicates "
        "the best compression ratio (slowest).",
    )


class Zarr2CompressorZlib(_Zarr2Compressor):
    """Zlib compressor specification."""

    id: Literal["zlib"] = "zlib"
    level: Annotated[int, Interval(ge=0, le=9)] = Field(
        default=1,
        description="Specifies the zlib compression level to use. "
        "Level 0 indicates no compression (fastest), while level 9 indicates the "
        "best compression ratio (slowest).",
    )


class Zarr2CompressorZstd(_Zarr2Compressor):
    """Zstd compressor specification."""

    id: Literal["zstd"] = "zstd"
    level: Annotated[int, Interval(ge=-131072, le=22)] = Field(
        default=1,
        description="Specifies the zstd compression level to use. "
        "A higher compression level provides improved density but reduced "
        "compression speed.",
    )


Zarr2Compressor: TypeAlias = Annotated[
    (
        Zarr2CompressorBlosc
        | Zarr2CompressorBz2
        | Zarr2CompressorZlib
        | Zarr2CompressorZstd
    ),
    Field(discriminator="id"),
]


class Zarr2Codec(CodecBase):
    """Zarr2 codec specification."""

    driver: Literal["zarr"] = "zarr"
    compressor: Zarr2Compressor | None = Field(
        default=None,
        description=(
            "Specifies the chunk compression method."
            "Specifying null disables compression. When creating a new array, "
            'if not specified, the default compressor of {"id": "blosc"} is used.'
        ),
    )
    filters: Literal[None] = Field(
        default=None,
        description="When encoding chunk, filters are applied before the compressor. "
        "Currently, filters are not supported..",
    )
