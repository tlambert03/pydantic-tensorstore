"""Zarr3 driver specification for Zarr v3 format."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    NonNegativeInt,
    PositiveInt,
)

from pydantic_tensorstore._types import DataType
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


def _str_to_codec(v: str) -> dict[str, Any]:
    if isinstance(v, str):
        return {"name": v}
    return v


#  TODO
SingleCodec: TypeAlias = Annotated[Any, BeforeValidator(_str_to_codec)]
CodecChain: TypeAlias = list[SingleCodec]


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

    codecs: CodecChain | None = Field(
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
