"""Neuroglancer Precomputed driver specification."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec
from pydantic_tensorstore.kvstore import KvStore  # noqa: TC001


class NeuroglancerPrecomputedSpec(ChunkedTensorStoreKvStoreAdapterSpec):
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

    model_config: ClassVar = {"extra": "forbid"}

    driver: Literal["neuroglancer_precomputed"] = Field(
        default="neuroglancer_precomputed",
        description="Neuroglancer Precomputed driver identifier",
    )

    kvstore: KvStore = Field(
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

    def get_effective_path(self) -> str:
        """Get the effective storage path."""
        if isinstance(self.kvstore, dict):
            kvstore_path = self.kvstore.get("path", "")
        else:
            kvstore_path = getattr(self.kvstore, "path", "")

        if not kvstore_path:
            return self.path
        kvstore_path = str(kvstore_path)
        if not self.path:
            return kvstore_path

        return f"{kvstore_path.rstrip('/')}/{self.path.lstrip('/')}"
