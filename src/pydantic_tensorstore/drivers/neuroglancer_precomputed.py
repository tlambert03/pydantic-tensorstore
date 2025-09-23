"""Neuroglancer Precomputed driver specification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec


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

    driver: Literal["neuroglancer_precomputed"] = "neuroglancer_precomputed"

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
