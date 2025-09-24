"""Neuroglancer Precomputed driver specification."""

from typing import Annotated, Any, Literal

from annotated_types import Interval
from pydantic import Field

from pydantic_tensorstore.core.codec import CodecBase
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


class NeuroglancerPrecomputedCodec(CodecBase):
    """Neuroglancer Precomputed codec specification."""

    driver: Literal["neuroglancer_precomputed"] = "neuroglancer_precomputed"
    encoding: Literal["raw", "jpeg", "png", "compressed_segmentation"] | None = Field(
        default=None,
        description="Specifies the chunk encoding.  Required when creating a new scale",
    )
    jpeg_quality: Annotated[int, Interval(ge=0, le=100)] | None = Field(
        default=None,
        description="JPEG quality (1-100). Only used if encoding is 'jpeg'.",
    )
    png_level: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None,
        description="PNG compression level (0-9). Only used if encoding is 'png'.",
    )
    shard_data_encoding: Literal["raw", "gzip"] | None = Field(
        default=None,
        description="Additional data compression when using the sharded format.",
    )
