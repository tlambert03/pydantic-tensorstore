"""Neuroglancer Precomputed driver specification."""

from typing import Any, Literal

from pydantic import Field

from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec


class NeuroglancerPrecomputedSpec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Neuroglancer Precomputed format driver specification."""

    driver: Literal["neuroglancer_precomputed"] = "neuroglancer_precomputed"

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
