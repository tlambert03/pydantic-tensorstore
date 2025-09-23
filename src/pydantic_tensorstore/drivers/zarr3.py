"""Zarr3 driver specification for Zarr v3 format."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator

from pydantic_tensorstore._types import DataType, JsonObject
from pydantic_tensorstore.core.spec import BaseDriverSpec
from pydantic_tensorstore.kvstore import KvStoreSpec  # noqa: TC001


class Zarr3Metadata(BaseDriverSpec):
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

    shape: list[int] = Field(
        description="Array shape",
    )

    data_type: DataType | str = Field(
        description="Data type specification",
    )

    chunk_grid: dict[str, Any] = Field(
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

    @field_validator("shape", mode="before")
    @classmethod
    def validate_shape(cls, v: Any) -> list[int]:
        """Validate array shape."""
        if not isinstance(v, list):
            raise ValueError("shape must be a list")

        for i, dim_size in enumerate(v):
            if not isinstance(dim_size, int) or dim_size < 0:
                raise ValueError(f"Shape dimension {i} must be a non-negative integer")

        return v

    @field_validator("data_type", mode="before")
    @classmethod
    def validate_data_type(cls, v: Any) -> DataType | str:
        """Validate data type specification."""
        if isinstance(v, str):
            # Try to convert to DataType enum if possible
            try:
                return DataType(v)
            except ValueError:
                # Allow pass-through for Zarr3-specific data types
                return str(v)
        if isinstance(v, DataType):
            return v
        # Convert anything else to string
        return str(v)


class Zarr3Spec(BaseDriverSpec):
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

    kvstore: KvStoreSpec | JsonObject = Field(
        description="Key-value store for data storage",
    )

    path: str = Field(
        default="",
        description="Path within the kvstore for this array",
    )

    metadata: Zarr3Metadata | JsonObject | None = Field(
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
