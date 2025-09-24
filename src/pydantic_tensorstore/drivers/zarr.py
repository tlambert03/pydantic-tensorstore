"""Zarr driver specification for Zarr v2 format."""

import re
from typing import Annotated, Any, Literal, TypeAlias

from annotated_types import Interval
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

from pydantic_tensorstore.core.codec import CodecBase
from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec

# Pattern for basic types: <|>|b|i|u|f|c|m|M|S|U|V followed by number
BASIC_PATTERN = re.compile(r"^[<>|][biufcmMSUV]\d+$")
# Pattern for datetime/timedelta with units: <|>|[mM]8[units]
DATETIME_PATTERN = re.compile(r"^[<>|][mM]8\[[\w/]+\]$")


def _validate_simple_zarr2_dtype(obj: str) -> str:
    if BASIC_PATTERN.match(obj) or DATETIME_PATTERN.match(obj):
        return obj
    raise ValueError(
        f"Invalid Zarr v2 data type: '{obj}'. Must follow NumPy typestr format "
        f"(e.g., '<f8', '>i4', '|b1', '<M8[ns]')"
    )


def _validate_structured_zarr2_dtype(obj: list[Any]) -> list[Any]:
    """Validate Zarr v2 data type encoding.

    Supports both simple data types (NumPy typestr format) and structured data types.
    """

    # Structured data type validation
    def _validate_field(field: Any) -> None:
        if not isinstance(field, list) or len(field) < 2 or len(field) > 3:
            raise ValueError(
                f"Invalid field format: {field}. Must be [fieldname, datatype] "
                f"or [fieldname, datatype, shape]"
            )

        fieldname, datatype = field[0], field[1]

        if not isinstance(fieldname, str):
            raise ValueError(f"Field name must be string, got {type(fieldname)}")

        if isinstance(datatype, str):
            # Simple datatype - validate recursively
            _validate_simple_zarr2_dtype(datatype)
        elif isinstance(datatype, list):
            # Nested structured datatype
            for nested_field in datatype:
                _validate_field(nested_field)
        else:
            raise ValueError(
                f"Invalid datatype in field '{fieldname}': {datatype}. "
                f"Must be string or list"
            )

        # Validate optional shape
        if len(field) == 3:
            shape = field[2]
            if not isinstance(shape, list) or not all(
                isinstance(dim, int) and dim > 0 for dim in shape
            ):
                raise ValueError(
                    f"Invalid shape in field '{fieldname}': {shape}. "
                    f"Must be list of positive integers"
                )

    # Validate each field
    for field in obj:
        _validate_field(field)

    return obj


Zarr2SimpleDataType: TypeAlias = Annotated[
    str, AfterValidator(_validate_simple_zarr2_dtype)
]
Zarr2StructuredDataType: TypeAlias = Annotated[
    list[Any], AfterValidator(_validate_structured_zarr2_dtype)
]
Zarr2DataType: TypeAlias = Zarr2SimpleDataType | Zarr2StructuredDataType


class ZarrMetadata(BaseModel):
    """Zarr v2 metadata specification.

    Controls Zarr-specific format options like compression,
    chunk shapes, and array metadata.
    """

    zarr_format: Literal[2] = 2

    shape: list[NonNegativeInt] | None = Field(
        default=None,
        description=(
            "Array shape. Required when creating a new array "
            "if the `Schema.domain` is not otherwise specified."
        ),
    )

    chunks: list[PositiveInt] | None = Field(
        default=None,
        description="Chunk dimensions. Must have the same length as shape.",
    )

    dtype: Zarr2DataType | None = Field(
        default=None,
        description="Data type specification",
    )

    fill_value: Any = Field(
        default=None,
        description="Fill value for uninitialized chunks",
    )

    order: Literal["C", "F"] = Field(
        default="C",
        description="Memory layout order (C=row-major, F=column-major)",
    )

    compressor: "Zarr2Compressor | None" = Field(
        default=None,
        description="Compression configuration",
    )

    filters: Literal[None] = Field(
        default=None,
        description="Filter pipeline configuration (currently not supported)",
    )

    dimension_separator: Literal[".", "/"] = Field(
        default=".",
        description="Separator for dimension names in chunk keys",
    )

    @model_validator(mode="after")
    def _validate_chunk_shape_length(self) -> Self:
        """Validate that chunks length matches array shape length."""
        if self.shape is not None and self.chunks is not None:
            shape_len = len(self.shape)
            chunks_len = len(self.chunks)
            if shape_len != chunks_len:
                raise ValueError(
                    f"chunks length ({chunks_len}) must match "
                    f"shape length ({shape_len})"
                )
        return self


class Zarr2Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Zarr driver specification for Zarr v2 format."""

    driver: Literal["zarr"] = "zarr"

    metadata: ZarrMetadata | None = Field(
        default=None,
        description="Zarr metadata specification",
    )


class _Zarr2Compressor(BaseModel):
    """Base class for Zarr v2 compressor specifications.

    The id member identifies the compressor.
    The remaining members are specific to the compressor.
    """


class Zarr2CompressorBlosc(_Zarr2Compressor):
    """Blosc compressor specification."""

    id: Literal["blosc"] = "blosc"
    cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] = Field(
        default="lz4",
        description="Compression algorithm",
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


def _str_to_compressor(v: str) -> dict[str, Any]:
    """A plain string is equivalent to an object with the string as its id.

    For example, "blosc" is equivalent to {"id": "blosc"}.
    """
    if isinstance(v, str):
        return {"id": v}
    return v


Zarr2Compressor: TypeAlias = Annotated[
    Zarr2CompressorBlosc
    | Zarr2CompressorBz2
    | Zarr2CompressorZlib
    | Zarr2CompressorZstd,
    Field(discriminator="id"),
    BeforeValidator(_str_to_compressor),
]
ZarrMetadata.model_rebuild()


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
