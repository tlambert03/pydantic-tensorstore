"""Neuroglancer Precomputed driver specification."""

from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

from annotated_types import Interval
from pydantic import AfterValidator, Field, NonNegativeInt, PositiveInt, model_validator

from pydantic_tensorstore._core.base import TensorStoreModel
from pydantic_tensorstore._core.codec import CodecBase
from pydantic_tensorstore._core.spec import ChunkedTensorStoreKvStoreAdapterSpec
from pydantic_tensorstore._kvstore.neuroglancer_uint64_sharded import (
    NeuroglancerShardingSpec,
)
from pydantic_tensorstore._types import DataType

__all__ = [
    "VALID_NEUROGLANCER_DTYPES",
    "NeuroglancerDataType",
    "NeuroglancerEncoding",
    "NeuroglancerMultiscaleMetadata",
    "NeuroglancerPrecomputedCodec",
    "NeuroglancerPrecomputedSpec",
    "NeuroglancerScaleMetadata",
    "NeuroglancerShardingSpec",
]

VALID_NEUROGLANCER_DTYPES: set[DataType] = {
    DataType.UINT8,
    DataType.UINT16,
    DataType.UINT32,
    DataType.UINT64,
    DataType.FLOAT32,
}


def _validate_ng_dtype(v: DataType) -> DataType:
    if v not in VALID_NEUROGLANCER_DTYPES:
        raise ValueError(
            f"Invalid Neuroglancer data type: {v}. "
            f"Must be one of {VALID_NEUROGLANCER_DTYPES}"
        )
    return v


NeuroglancerDataType: TypeAlias = Annotated[
    DataType, AfterValidator(_validate_ng_dtype)
]
NeuroglancerEncoding: TypeAlias = Literal[
    "raw", "jpeg", "png", "compressed_segmentation"
]
_XYZ = tuple[int, int, int]


class NeuroglancerMultiscaleMetadata(TensorStoreModel):
    """Scale-independent metadata (from the `info` file)."""

    type: Literal["image", "segmentation"] | None = Field(
        default=None,
        description="Volume type; used by Neuroglancer to pick the layer type. "
        "Required when creating a new multiscale volume.",
    )
    data_type: NeuroglancerDataType | None = Field(
        default=None,
        description="Data type. Required when creating a new multiscale volume.",
    )
    num_channels: PositiveInt | None = Field(
        default=None,
        description="Number of channels. Required when creating a new volume.",
    )


class NeuroglancerScaleMetadata(TensorStoreModel):
    """Per-scale metadata (from the `info` file)."""

    key: str | None = Field(
        default=None,
        description="Scale key relative to `path`. Default when creating: "
        '`"<xres>_<yres>_<zres>"`.',
    )
    size: tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt] | None = Field(
        default=None,
        description="Voxel dimensions (XYZ). Required when creating a new scale if "
        "`Schema.domain` is not specified.",
    )
    voxel_offset: _XYZ | None = Field(
        default=None,
        description="Voxel origin (XYZ). Requires `size`. Default: `[0, 0, 0]`.",
    )
    chunk_size: tuple[PositiveInt, PositiveInt, PositiveInt] | None = Field(
        default=None, description="Chunk dimensions (XYZ)."
    )
    resolution: tuple[float, float, float] | None = Field(
        default=None, description="Voxel size in nanometers (XYZ)."
    )
    encoding: NeuroglancerEncoding | None = Field(
        default=None,
        description="Chunk encoding. Required when creating a new scale.",
    )
    jpeg_quality: Annotated[int, Interval(ge=0, le=100)] | None = Field(
        default=None,
        description="JPEG quality (0-100), only for `jpeg` encoding. Default: `75`.",
    )
    png_level: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None,
        description="PNG compression level (0-9), only for `png` encoding.",
    )
    compressed_segmentation_block_size: tuple[float, float, float] | None = Field(
        default=None,
        description="Block size (XYZ) for `compressed_segmentation` encoding. "
        "Must not be specified with any other encoding.",
    )
    sharding: NeuroglancerShardingSpec | None = Field(
        default=None,
        description="Sharded format spec; `null` (the default) means unsharded.",
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Cross-field checks that tensorstore also enforces."""
        if self.voxel_offset is not None and self.size is None:
            raise ValueError("voxel_offset cannot be specified without size")
        if (
            self.compressed_segmentation_block_size is not None
            and self.encoding is not None
            and self.encoding != "compressed_segmentation"
        ):
            raise ValueError(
                "compressed_segmentation_block_size requires encoding "
                "'compressed_segmentation'"
            )
        return self


class NeuroglancerPrecomputedSpec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Neuroglancer Precomputed format driver specification."""

    driver: Literal["neuroglancer_precomputed"] = "neuroglancer_precomputed"
    scale_index: NonNegativeInt | None = Field(
        default=None,
        description="Zero-based index of the scale to open or create.",
    )
    multiscale_metadata: NeuroglancerMultiscaleMetadata | None = None
    scale_metadata: NeuroglancerScaleMetadata | None = None


class NeuroglancerPrecomputedCodec(CodecBase):
    """Neuroglancer Precomputed codec specification."""

    driver: Literal["neuroglancer_precomputed"] = "neuroglancer_precomputed"
    encoding: NeuroglancerEncoding | None = Field(
        default=None,
        description="Chunk encoding. Required when creating a new scale.",
    )
    jpeg_quality: Annotated[int, Interval(ge=0, le=100)] | None = Field(
        default=None,
        description="JPEG quality (0-100), only for `jpeg` encoding. Default: `75`.",
    )
    png_level: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None,
        description="PNG compression level (0-9), only for `png` encoding.",
    )
    shard_data_encoding: Literal["raw", "gzip"] | None = Field(
        default=None,
        description="Additional data compression when using the sharded format. "
        'Default: `"gzip"` for raw/compressed_segmentation, `"raw"` for jpeg.',
    )
