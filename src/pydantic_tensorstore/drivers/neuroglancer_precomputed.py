"""Neuroglancer Precomputed driver specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator

from pydantic_tensorstore.core.spec import BaseDriverSpec

if TYPE_CHECKING:
    from pydantic_tensorstore.kvstore.base import KvStoreSpec
    from pydantic_tensorstore.types.common import JsonObject


class NeuroglancerPrecomputedSpec(BaseDriverSpec):
    """Neuroglancer Precomputed format driver specification.

    Supports the Neuroglancer Precomputed format used for large-scale
    visualization of volumetric data in the Neuroglancer viewer.

    Attributes
    ----------
        driver: Must be "neuroglancer_precomputed"
        kvstore: Key-value store for data storage
        path: Path within the kvstore
        scale_index: Index of the scale to use
        multiscale_metadata: Metadata for multiscale pyramid
        scale_metadata: Metadata for specific scale level

    Example:
        >>> spec = NeuroglancerPrecomputedSpec(
        ...     driver="neuroglancer_precomputed",
        ...     kvstore={"driver": "file", "path": "/data/precomputed/"},
        ...     scale_index=0,
        ... )
    """

    model_config = {"extra": "forbid"}

    driver: Literal["neuroglancer_precomputed"] = Field(
        default="neuroglancer_precomputed",
        description="Neuroglancer Precomputed driver identifier",
    )

    kvstore: KvStoreSpec | JsonObject = Field(
        description="Key-value store for data storage",
    )

    path: str = Field(
        default="",
        description="Path within the kvstore",
    )

    scale_index: int | None = Field(
        default=None,
        description="Index of the scale to use from the multiscale pyramid",
        ge=0,
    )

    multiscale_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Multiscale metadata configuration",
    )

    scale_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Scale-specific metadata",
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

    def get_driver_kind(self) -> str:
        """Get the driver kind."""
        return "tensorstore"

    def get_effective_path(self) -> str:
        """Get the effective storage path."""
        if isinstance(self.kvstore, dict):
            kvstore_path = self.kvstore.get("path", "")
        else:
            kvstore_path = getattr(self.kvstore, "path", "")

        if not kvstore_path:
            return self.path
        if not self.path:
            return kvstore_path

        return f"{kvstore_path.rstrip('/')}/{self.path.lstrip('/')}"
