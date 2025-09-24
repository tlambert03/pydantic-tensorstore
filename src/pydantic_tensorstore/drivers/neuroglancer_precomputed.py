"""Neuroglancer Precomputed driver specification."""

from typing import Annotated, ClassVar, Literal, TypeAlias

from annotated_types import Interval
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

from pydantic_tensorstore._types import DataType
from pydantic_tensorstore.core.codec import CodecBase
from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec

VALID_NEUROGLANCER_DTYPES: set[DataType] = {
    DataType.UINT8,
    DataType.UINT16,
    DataType.UINT32,
    DataType.UINT64,
    DataType.FLOAT32,
}

NeuroglancerDataType: TypeAlias = Annotated[
    DataType,
    AfterValidator(
        lambda v: v in VALID_NEUROGLANCER_DTYPES
        or ValueError(
            f"Invalid Neuroglancer data type: {v}. "
            "Must be one of {VALID_NEUROGLANCER_DTYPES}"
        )
    ),
]


class NeuroglancerMultiscaleMetadata(BaseModel):
    """Neuroglancer Precomputed multiscale metadata specification."""

    type: Literal["image", "segmentation"] | None = Field(
        default=None, description="Volume type specification"
    )
    data_type: NeuroglancerDataType | None = Field(
        default=None, description="Data type specification"
    )
    num_channels: PositiveInt | None = Field(
        default=None, description="Number of channels"
    )


class NeuroglancerShardingSpec(BaseModel):
    """Neuroglancer sharding specification."""

    model_config: ClassVar[ConfigDict] = ConfigDict(serialize_by_alias=True)

    type_: Literal["neuroglancer_uint64_sharded_v1"] = Field(
        default="neuroglancer_uint64_sharded_v1", alias="@type"
    )
    preshift_bits: Annotated[int, Interval(ge=0, le=64)] = Field(
        description="Number of low-order bits of the chunk ID that do not contribute "
        "to the hashed chunk ID.",
    )
    hash: Literal["identity", "murmurhash3_x86_128"] = Field(
        description="Hash function for sharding"
    )
    minishard_bits: Annotated[int, Interval(ge=0, le=64)] = Field(
        description="Number of bits of the hashed chunk ID that "
        "determine the minishard number."
    )
    shard_bits: Annotated[int, Interval(ge=0, le=64)] = Field(
        description="Number of bits of the hashed chunk ID that "
        "determine the shard number."
    )
    minishard_index_encoding: Literal["gzip", "raw"] | None = Field(
        default=None,
        description="Specifies the encoding of the minishard index. "
        "Normally 'gzip' is a good choice.",
    )
    data_encoding: Literal["gzip", "raw"] | None = Field(
        default=None,
        description="Specifies the encoding of the chunk data. "
        "Normally 'gzip' is a good choice, unless the volume uses jpeg encoding.",
    )


class NeuroglancerScaleMetadata(BaseModel):
    """Neuroglancer Precomputed scale metadata specification."""

    key: str | None = Field(default=None, description="Scale identifier string")
    size: tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt] | None = Field(
        default=None, description="3D volume dimensions [x, y, z]"
    )
    voxel_offset: tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt] | None = Field(
        default_factory=lambda: (0, 0, 0), description="3D origin coordinates [x, y, z]"
    )
    chunk_size: tuple[PositiveInt, PositiveInt, PositiveInt] | None = Field(
        default=None, description="3D chunk dimensions [x, y, z]"
    )
    resolution: tuple[float, float, float] | None = Field(
        default=None, description="Voxel size in nanometers [x, y, z]"
    )
    encoding: Literal["raw", "jpeg", "png", "compressed_segmentation"] | None = Field(
        default=None, description="Chunk encoding type"
    )
    jpeg_quality: Annotated[int, Interval(ge=0, le=100)] | None = Field(
        default=None,
        description=(
            "JPEG compression quality (0-100). Only used if encoding is 'jpeg'."
        ),
    )
    png_level: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None,
        description="PNG compression level (0-9). Only used if encoding is 'png'.",
    )
    compressed_segmentation_block_size: tuple[float, float, float] | None = Field(
        default=None,
        description="Block size for compressed segmentation encoding [x, y, z]",
    )
    sharding: NeuroglancerShardingSpec | None = Field(
        default=None, description="Optional sharding configuration"
    )

    @model_validator(mode="after")
    def _validate_dimensions(self) -> Self:
        """Validate that all dimension arrays have length 3."""
        if len(self.size) != 3:
            raise ValueError(f"size must have length 3, got {len(self.size)}")
        if len(self.voxel_offset) != 3:
            raise ValueError(
                f"voxel_offset must have length 3, got {len(self.voxel_offset)}"
            )
        if len(self.chunk_size) != 3:
            raise ValueError(
                f"chunk_size must have length 3, got {len(self.chunk_size)}"
            )
        if len(self.resolution) != 3:
            raise ValueError(
                f"resolution must have length 3, got {len(self.resolution)}"
            )
        if self.compressed_segmentation_block_size is not None:
            if len(self.compressed_segmentation_block_size) != 3:
                raise ValueError(
                    f"compressed_segmentation_block_size must have length 3, "
                    f"got {len(self.compressed_segmentation_block_size)}"
                )
        return self

    @model_validator(mode="after")
    def _validate_encoding_parameters(self) -> Self:
        """Validate encoding-specific parameters."""
        if self.encoding == "jpeg" and self.jpeg_quality is None:
            raise ValueError("jpeg_quality is required when encoding is 'jpeg'")
        if self.encoding == "png" and self.png_level is None:
            raise ValueError("png_level is required when encoding is 'png'")
        if (
            self.encoding == "compressed_segmentation"
            and self.compressed_segmentation_block_size is None
        ):
            raise ValueError(
                "compressed_segmentation_block_size is required when encoding is "
                "'compressed_segmentation'"
            )
        return self


class NeuroglancerPrecomputedSpec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Neuroglancer Precomputed format driver specification."""

    driver: Literal["neuroglancer_precomputed"] = "neuroglancer_precomputed"

    scale_index: NonNegativeInt | None = Field(
        default=None,
        description="Zero-based index of the scale to use from the multiscale pyramid",
    )

    multiscale_metadata: NeuroglancerMultiscaleMetadata | None = Field(
        default=None,
        description="Multiscale metadata configuration",
    )

    scale_metadata: NeuroglancerScaleMetadata | None = Field(
        default=None,
        description="Scale-specific metadata",
    )


class NeuroglancerPrecomputedCodec(CodecBase):
    """Neuroglancer Precomputed codec specification."""

    driver: Literal["neuroglancer_precomputed"] = "neuroglancer_precomputed"
    encoding: Literal["raw", "jpeg", "png", "compressed_segmentation"] | None = Field(
        default=None,
        description="Specifies the chunk encoding. Required when creating a new scale.",
    )
    jpeg_quality: Annotated[int, Interval(ge=0, le=100)] | None = Field(
        default=None,
        description=(
            "JPEG compression quality (0-100). Only used if encoding is 'jpeg'."
        ),
    )
    png_level: Annotated[int, Interval(ge=0, le=9)] | None = Field(
        default=None,
        description="PNG compression level (0-9). Only used if encoding is 'png'.",
    )
    shard_data_encoding: Literal["raw", "gzip"] | None = Field(
        default=None,
        description="Additional data compression when using the sharded format.",
    )

    @model_validator(mode="after")
    def _validate_encoding_parameters(self) -> Self:
        """Validate encoding-specific parameters."""
        if self.encoding == "jpeg" and self.jpeg_quality is None:
            raise ValueError("jpeg_quality is required when encoding is 'jpeg'")
        if self.encoding == "png" and self.png_level is None:
            raise ValueError("png_level is required when encoding is 'png'")
        return self
