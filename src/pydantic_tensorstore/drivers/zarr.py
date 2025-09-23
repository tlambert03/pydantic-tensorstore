"""Zarr driver specification for Zarr v2 format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, field_validator

from pydantic_tensorstore.core.spec import BaseDriverSpec

if TYPE_CHECKING:
    from pydantic_tensorstore.kvstore.base import KvStoreSpec
    from pydantic_tensorstore.types.common import JsonObject


class ZarrMetadata(BaseDriverSpec):
    """Zarr metadata specification.

    Controls Zarr-specific format options like compression,
    chunk shapes, and array metadata.
    """

    model_config: ClassVar = {"extra": "allow"}  # Allow additional zarr metadata fields

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

    def get_driver_kind(self) -> str:
        """Get the driver kind."""
        return "metadata"


class ZarrSpec(BaseDriverSpec):
    """Zarr driver specification for Zarr v2 format.

    Zarr is a format for storing chunked, compressed arrays.
    This spec supports the Zarr v2 specification.

    Attributes
    ----------
        driver: Must be "zarr"
        kvstore: Key-value store for data storage
        path: Path within the kvstore for this array
        metadata: Zarr-specific metadata and options

    Example:
        >>> spec = ZarrSpec(
        ...     driver="zarr",
        ...     kvstore={"driver": "memory"},
        ...     metadata={
        ...         "chunks": [64, 64],
        ...         "compressor": {"id": "blosc", "cname": "lz4"},
        ...     },
        ... )
        >>> # With file storage
        >>> spec = ZarrSpec(
        ...     driver="zarr",
        ...     kvstore={"driver": "file", "path": "/data/arrays/"},
        ...     path="my_array.zarr",
        ... )
    """

    model_config: ClassVar = {"extra": "forbid"}

    driver: Literal["zarr"] = Field(
        default="zarr",
        description="Zarr driver identifier",
    )

    kvstore: KvStoreSpec | JsonObject = Field(
        description="Key-value store for data storage",
    )

    path: str = Field(
        default="",
        description="Path within the kvstore for this array",
    )

    metadata: ZarrMetadata | JsonObject | None = Field(
        default=None,
        description="Zarr metadata specification",
    )

    recheck_cached_data: bool | None = Field(
        default=None,
        description="Whether to recheck cached data",
    )

    recheck_cached_metadata: bool | None = Field(
        default=None,
        description="Whether to recheck cached metadata",
    )

    @field_validator("kvstore", mode="before")
    @classmethod
    def validate_kvstore(cls, v: Any) -> KvStoreSpec | JsonObject:
        """Validate kvstore specification."""
        if isinstance(v, dict):
            if "driver" not in v:
                raise ValueError("kvstore must specify a driver")
            return v

        return v

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, v: Any) -> str:
        """Validate array path."""
        if not isinstance(v, str):
            raise ValueError("path must be a string")
        return v

    def get_driver_kind(self) -> str:
        """Get the driver kind."""
        return "tensorstore"

    def get_effective_path(self) -> str:
        """Get the effective storage path combining kvstore and array path."""
        if isinstance(self.kvstore, dict):
            kvstore_path = self.kvstore.get("path", "")
        else:
            kvstore_path = getattr(self.kvstore, "path", "")

        if not kvstore_path:
            return self.path
        if not self.path:
            return kvstore_path

        # Combine paths appropriately
        if kvstore_path.endswith("/") or self.path.startswith("/"):
            return kvstore_path + self.path
        else:
            return f"{kvstore_path}/{self.path}"

    def get_zarr_metadata_defaults(self) -> dict[str, Any]:
        """Get default Zarr metadata for this array."""
        defaults = {
            "zarr_format": 2,
            "order": "C",
            "dimension_separator": ".",
        }

        if self.metadata:
            if isinstance(self.metadata, dict):
                defaults.update(self.metadata)
            else:
                # It's a ZarrMetadata object
                defaults.update(self.metadata.model_dump(exclude_unset=True))

        return defaults
