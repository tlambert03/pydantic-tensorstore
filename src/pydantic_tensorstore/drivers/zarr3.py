"""Zarr3 driver specification for Zarr v3 format."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from annotated_types import Ge
from pydantic import BaseModel, Field

from pydantic_tensorstore._types import DataType
from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec
from pydantic_tensorstore.kvstore import KvStore  # noqa: TC001

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
    """Zarr3 driver specification for Zarr v3 format.

    Zarr v3 is the next generation of the Zarr format, featuring
    improved performance, sharding, and enhanced codec support.

    Attributes
    ----------
        driver: Must be "zarr3"
        kvstore: Key-value store for data storage
        path: Path within the kvstore for this array
        metadata: Zarr v3 metadata specification

    Example:
        >>> spec = Zarr3Spec(
        ...     driver="zarr3",
        ...     kvstore={"driver": "memory"},
        ...     metadata={
        ...         "shape": [1000, 2000],
        ...         "data_type": "float32",
        ...         "chunk_grid": {
        ...             "name": "regular",
        ...             "configuration": {"chunk_shape": [100, 200]},
        ...         },
        ...     },
        ... )
    """

    model_config: ClassVar = {"extra": "forbid"}

    driver: Literal["zarr3"] = Field(
        default="zarr3",
        description="Zarr3 driver identifier",
    )

    kvstore: KvStore = Field(
        description="Key-value store for data storage",
    )

    path: str = Field(
        default="",
        description="Path within the kvstore for this array",
    )

    metadata: Zarr3Metadata | None = Field(
        default=None,
        description="Zarr v3 metadata specification",
    )

    def get_effective_path(self) -> str:
        """Get the effective storage path."""
        if isinstance(self.kvstore, dict):
            kvstore_path = str(self.kvstore.get("path", ""))
        else:
            kvstore_path = str(getattr(self.kvstore, "path", ""))

        if not kvstore_path:
            return self.path
        if not self.path:
            return kvstore_path

        return f"{kvstore_path.rstrip('/')}/{self.path.lstrip('/')}"

    def get_zarr3_defaults(self) -> dict[str, Any]:
        """Get default Zarr v3 configuration."""
        defaults = {
            "zarr_format": 3,
            "node_type": "array",
            "chunk_key_encoding": {"name": "default", "separator": "/"},
            "attributes": {},
        }

        if self.metadata:
            if isinstance(self.metadata, dict):
                defaults.update(self.metadata)
            else:
                defaults.update(self.metadata.model_dump(exclude_unset=True))

        return defaults
