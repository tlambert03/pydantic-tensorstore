"""Zarr driver specification for Zarr v2 format."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Self, TypeAlias

from annotated_types import Interval
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from pydantic_tensorstore._core.base import TensorStoreModel, since
from pydantic_tensorstore._core.codec import CodecBase
from pydantic_tensorstore._core.spec import ChunkedTensorStoreKvStoreAdapterSpec
from pydantic_tensorstore._types import DTYPE_SINCE

# Pattern for basic types: <|>|b|i|u|f|c|m|M|S|U|V followed by number
BASIC_PATTERN = re.compile(r"^[<>|][biufcmMSUV]\d+$")
# Pattern for datetime/timedelta with units: <|>|[mM]8[units]
DATETIME_PATTERN = re.compile(r"^[<>|][mM]8\[[\w/]+\]$")
# tensorstore extensions (little endian): bfloat16, float8_*, float4_*, int2/int4
EXTENSION_DTYPES = frozenset(
    {
        "bfloat16",
        "float8_e3m4",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e4m3b11fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float4_e2m1fn",
        "int2",
        "int4",
    }
)


def _validate_simple_zarr2_dtype(obj: str) -> str:
    if (
        BASIC_PATTERN.match(obj)
        or DATETIME_PATTERN.match(obj)
        or obj in EXTENSION_DTYPES
    ):
        return obj
    raise ValueError(
        f"Invalid Zarr v2 data type: '{obj}'. Must follow NumPy typestr format "
        f"(e.g., '<f8', '>i4', '|b1', '<M8[ns]') or be one of "
        f"{sorted(EXTENSION_DTYPES)}"
    )


def _validate_structured_zarr2_dtype(obj: list[Any]) -> list[Any]:
    """Validate a Zarr v2 structured data type encoding."""

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
            _validate_simple_zarr2_dtype(datatype)
        elif isinstance(datatype, list):
            for nested_field in datatype:
                _validate_field(nested_field)
        else:
            raise ValueError(
                f"Invalid datatype in field '{fieldname}': {datatype}. "
                f"Must be string or list"
            )

        if len(field) == 3:
            shape = field[2]
            if not isinstance(shape, list) or not all(
                isinstance(dim, int) and dim > 0 for dim in shape
            ):
                raise ValueError(
                    f"Invalid shape in field '{fieldname}': {shape}. "
                    f"Must be list of positive integers"
                )

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


class _Zarr2Compressor(TensorStoreModel):
    """Base class for Zarr v2 compressor specifications.

    The id member identifies the compressor.
    The remaining members are specific to the compressor.
    """


class Zarr2CompressorBlosc(_Zarr2Compressor):
    """Blosc compressor specification."""

    id: Literal["blosc"] = "blosc"
    cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] | None = Field(
        default=None, description='Compression algorithm. Default: `"lz4"`.'
    )
    clevel: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None,
        description="Blosc compression level; higher is slower but smaller. "
        "Default: `5`.",
    )
    shuffle: Literal[-1, 0, 1, 2] | None = Field(
        default=None,
        description="Shuffle filter: 0 none, 1 byte-wise, 2 bit-wise, -1 automatic "
        "(bit-wise for 1-byte elements, otherwise byte-wise). Default: `-1`.",
    )
    blocksize: NonNegativeInt | None = Field(
        default=None,
        description="Block size in bytes. Default: `0` (automatic).",
    )


class Zarr2CompressorBz2(_Zarr2Compressor):
    """Bz2 compressor specification."""

    id: Literal["bz2"] = "bz2"
    level: Annotated[int, Interval(ge=1, le=9)] | None = Field(
        default=None,
        description="bzip2 buffer size/compression level; 1 is fastest, 9 is the "
        "best ratio. Default: `1`.",
    )


class Zarr2CompressorZlib(_Zarr2Compressor):
    """Zlib compressor specification (zlib or gzip header)."""

    id: Literal["zlib", "gzip"] = "zlib"
    level: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None,
        description="zlib compression level; 0 is no compression, 9 is the best "
        "ratio. Default: `1`.",
    )


class Zarr2CompressorZstd(_Zarr2Compressor):
    """Zstd compressor specification."""

    id: Literal["zstd"] = "zstd"
    level: Annotated[int, Interval(ge=-131072, le=22)] | None = Field(
        default=None,
        description="zstd compression level; higher is denser but slower. "
        "Default: `1`.",
    )


def _str_to_compressor(v: Any) -> Any:
    """Convenience: a plain string `"blosc"` becomes `{"id": "blosc"}`.

    Note that tensorstore itself only accepts the object form.
    """
    return {"id": v} if isinstance(v, str) else v


Zarr2Compressor: TypeAlias = Annotated[
    Zarr2CompressorBlosc
    | Zarr2CompressorBz2
    | Zarr2CompressorZlib
    | Zarr2CompressorZstd,
    Field(discriminator="id"),
    BeforeValidator(_str_to_compressor),
]


class Zarr2Metadata(TensorStoreModel):
    """Zarr v2 `.zarray` metadata; all members optional."""

    zarr_format: Literal[2] | None = None
    shape: list[NonNegativeInt] | None = Field(
        default=None,
        description="Array shape. Required when creating a new array "
        "if the `Schema.domain` is not otherwise specified.",
    )
    chunks: list[PositiveInt] | None = Field(
        default=None,
        description="Chunk dimensions. Must have the same length as shape.",
    )
    dtype: Zarr2DataType | None = Field(
        default=None,
        description="Scalar or structured data type.",
        json_schema_extra=since(DTYPE_SINCE),
    )
    fill_value: Any = Field(
        default=None, description="Fill value for uninitialized chunks"
    )
    order: Literal["C", "F"] | None = Field(
        default=None,
        description='Memory layout of encoded chunks. Default: `"C"`.',
    )
    compressor: Zarr2Compressor | None = Field(
        default=None,
        description="Chunk compressor. `null` disables compression. "
        'Default when creating: `{"id": "blosc"}`.',
    )
    filters: Literal[None] = Field(
        default=None, description="Filters are not supported; must be `null`."
    )
    dimension_separator: Literal[".", "/"] | None = Field(
        default=None,
        description='Separator for chunk keys. Default: `"."`.',
    )

    @model_validator(mode="after")
    def _validate_chunk_shape_length(self) -> Self:
        """Validate that chunks length matches array shape length."""
        if self.shape is not None and self.chunks is not None:
            if len(self.shape) != len(self.chunks):
                raise ValueError(
                    f"chunks length ({len(self.chunks)}) must match "
                    f"shape length ({len(self.shape)})"
                )
        return self


class Zarr2Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Zarr driver specification for Zarr v2 format."""

    driver: Literal["zarr", "zarr2"] = Field(
        default="zarr", json_schema_extra=since({"zarr2": "0.1.75"})
    )
    field: str | None = Field(
        default=None,
        description="Name of field to open. Must be specified if the metadata.dtype "
        "specified in the array metadata has more than one field.",
    )
    open_as_void: bool | None = Field(
        default=None,
        description="Open the array as raw bytes with an extra byte dimension. "
        "Cannot be combined with `field`. Default: `false`.",
        json_schema_extra=since("0.1.81"),
    )
    metadata: Zarr2Metadata | None = None
    metadata_key: str | None = Field(
        default=None,
        description='Key storing the array metadata. Default: `".zarray"`. '
        "A non-default value breaks compatibility with other zarr implementations.",
    )
    key_encoding: Literal[".", "/"] | None = Field(
        default=None,
        description="Encoding of chunk indices into keys.",
        deprecated="Deprecated. Equivalent to specifying metadata.dimension_separator.",
    )

    @model_validator(mode="after")
    def _validate_metadata(self) -> Self:
        """Cross-field checks that tensorstore also enforces."""
        if self.field is not None and self.open_as_void:
            raise ValueError("`field` and `open_as_void` cannot both be specified.")
        if self.metadata is not None and isinstance(self.metadata.dtype, list):
            field_names = [f[0] for f in self.metadata.dtype]
            if len(field_names) > 1 and not self.field and not self.open_as_void:
                raise ValueError(
                    "`field` must be specified if the metadata.dtype specified in "
                    "the array metadata has more than one field."
                )
            if self.field is not None and self.field not in field_names:
                raise ValueError(
                    f"field '{self.field}' not found in metadata.dtype fields "
                    f"{field_names}"
                )
        return self


class Zarr2Codec(CodecBase):
    """Zarr2 codec specification."""

    driver: Literal["zarr"] = "zarr"
    compressor: Zarr2Compressor | None = Field(
        default=None,
        description="Chunk compressor. `null` disables compression. "
        'Default when creating: `{"id": "blosc"}`.',
    )
    filters: Literal[None] = Field(
        default=None, description="Filters are not supported; must be `null`."
    )
