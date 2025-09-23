"""Zarr3 driver specification for Zarr v3 format."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from annotated_types import Ge
from pydantic import BaseModel, Field

from pydantic_tensorstore._types import DataType
from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec

Zarr3DataType: TypeAlias = Literal[
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
]


class _ZarrChunkConfiguration(BaseModel):
    chunk_shape: list[Annotated[int, Ge(1)]] | None = Field(
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


class Zarr3Metadata(ChunkedTensorStoreKvStoreAdapterSpec):
    """Zarr v3 metadata specification.

    Zarr v3 introduces new features like sharding, variable chunks,
    and improved codec pipelines.
    """

    model_config: ClassVar = {"extra": "allow"}

    zarr_format: Literal[3] = Field(
        default=3,
        description="Zarr format version",
    )

    node_type: Literal["array"] = Field(
        default="array",
        description="Node type (array for data arrays)",
    )

    shape: list[Annotated[int, Ge(0)]] = Field(
        description=(
            "Array shape. Required when creating a new array "
            "if the `Schema.domain` is not otherwise specified."
        ),
    )

    data_type: Zarr3DataType = Field(
        description="Data type specification",
    )

    chunk_grid: _ZarrChunkGrid | None = Field(
        default=None,
        description="Chunk grid specification",
    )

    chunk_key_encoding: dict[str, Any] = Field(
        default_factory=lambda: {"name": "default", "separator": "/"},
        description="Chunk key encoding configuration",
    )

    fill_value: int | float | str | bool | list[Any] | None = Field(
        default=None,
        description="Fill value for uninitialized chunks",
    )

    codecs: list[dict[str, Any]] | None = Field(
        default=None,
        description="Codec pipeline for compression and encoding",
    )

    dimension_names: list[str | None] | None = Field(
        default=None,
        description="Names for each dimension",
    )

    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined attributes",
    )


class Zarr3Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Zarr3 driver specification for Zarr v3 format."""

    driver: Literal["zarr3"] = "zarr3"

    metadata: Zarr3Metadata | None = Field(
        default=None,
        description="Zarr v3 metadata specification",
    )
