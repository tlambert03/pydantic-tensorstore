"""Zarr3 driver specification for Zarr v3 format."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from pydantic_tensorstore._core.base import TensorStoreModel, since
from pydantic_tensorstore._core.codec import CodecBase
from pydantic_tensorstore._core.spec import ChunkedTensorStoreKvStoreAdapterSpec
from pydantic_tensorstore._core.transform import _validate_labels
from pydantic_tensorstore._core.zarr3_codecs import (
    Zarr3BloscConfig,
    Zarr3BytesConfig,
    Zarr3CodecBlosc,
    Zarr3CodecBytes,
    Zarr3CodecChain,
    Zarr3CodecCRC32C,
    Zarr3CodecGzip,
    Zarr3CodecShardingIndexed,
    Zarr3CodecTranspose,
    Zarr3CodecZstd,
    Zarr3CRC32CConfig,
    Zarr3GzipConfig,
    Zarr3ShardingIndexedConfig,
    Zarr3SingleCodec,
    Zarr3TransposeConfig,
    Zarr3ZstdConfig,
)
from pydantic_tensorstore._types import DTYPE_SINCE, DataType

__all__ = [
    "VALID_ZARR3_DTYPES",
    "Zarr3Attributes",
    "Zarr3BloscConfig",
    "Zarr3BytesConfig",
    "Zarr3CRC32CConfig",
    "Zarr3ChunkConfiguration",
    "Zarr3ChunkGrid",
    "Zarr3ChunkKeyEncoding",
    "Zarr3ChunkKeyEncodingConfig",
    "Zarr3ChunkKeyEncodingDefault",
    "Zarr3ChunkKeyEncodingV2",
    "Zarr3Codec",
    "Zarr3CodecBlosc",
    "Zarr3CodecBytes",
    "Zarr3CodecCRC32C",
    "Zarr3CodecChain",
    "Zarr3CodecGzip",
    "Zarr3CodecShardingIndexed",
    "Zarr3CodecTranspose",
    "Zarr3CodecZstd",
    "Zarr3DataType",
    "Zarr3GzipConfig",
    "Zarr3Metadata",
    "Zarr3ShardingIndexedConfig",
    "Zarr3SingleCodec",
    "Zarr3Spec",
    "Zarr3TransposeConfig",
    "Zarr3ZstdConfig",
]

VALID_ZARR3_DTYPES: set[DataType] = {
    DataType.BOOL,
    DataType.INT2,
    DataType.INT4,
    DataType.INT8,
    DataType.INT16,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT8,
    DataType.UINT16,
    DataType.UINT32,
    DataType.UINT64,
    DataType.FLOAT8_E3M4,
    DataType.FLOAT8_E4M3FN,
    DataType.FLOAT8_E4M3FNUZ,
    DataType.FLOAT8_E4M3B11FNUZ,
    DataType.FLOAT8_E5M2,
    DataType.FLOAT8_E5M2FNUZ,
    DataType.FLOAT8_E8M0FNU,
    DataType.FLOAT4_E2M1FN,
    DataType.FLOAT16,
    DataType.BFLOAT16,
    DataType.FLOAT32,
    DataType.FLOAT64,
    DataType.COMPLEX64,
    DataType.COMPLEX128,
}


def _validate_zarr3_dtype(v: DataType) -> DataType:
    if v not in VALID_ZARR3_DTYPES:
        raise ValueError(
            f"Invalid Zarr3 data type: {v}. Must be one of {VALID_ZARR3_DTYPES}"
        )
    return v


Zarr3DataType: TypeAlias = (
    Annotated[DataType, AfterValidator(_validate_zarr3_dtype)]
    | Annotated[str, StringConstraints(pattern=r"^r\d+$")]
)
"""A zarr v3 data type: a supported `DataType`, or `r<N>` for N raw bits."""


class Zarr3ChunkConfiguration(TensorStoreModel):
    """Configuration for the regular chunk grid."""

    chunk_shape: list[PositiveInt] | None = Field(
        default=None,
        description="Chunk dimensions. Must have the same length as shape. If not "
        "specified when creating a new array, chosen according to "
        "`Schema.chunk_layout`.",
    )


class Zarr3ChunkGrid(TensorStoreModel):
    """Chunk grid specification."""

    name: Literal["regular"] = Field(
        default="regular",
        description="Chunk grid type (only 'regular' is supported)",
    )
    configuration: Zarr3ChunkConfiguration | None = None


class Zarr3ChunkKeyEncodingConfig(TensorStoreModel):
    """Chunk key encoding configuration."""

    separator: Literal["/", "."] | None = Field(
        default=None,
        description='Key separator. Default: `"/"` for `default`, `"."` for `v2`.',
    )


class Zarr3ChunkKeyEncodingDefault(TensorStoreModel):
    """Default chunk key encoding (`c/0/1` style)."""

    name: Literal["default"] = "default"
    configuration: Zarr3ChunkKeyEncodingConfig | None = None


class Zarr3ChunkKeyEncodingV2(TensorStoreModel):
    """Zarr v2 compatible chunk key encoding (`0.1` style)."""

    name: Literal["v2"] = "v2"
    configuration: Zarr3ChunkKeyEncodingConfig | None = None


Zarr3ChunkKeyEncoding: TypeAlias = Annotated[
    Zarr3ChunkKeyEncodingDefault | Zarr3ChunkKeyEncodingV2,
    Field(discriminator="name"),
]


class Zarr3Attributes(TensorStoreModel):
    """User-defined attributes; `dimension_units` is understood by tensorstore."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    dimension_units: list[str | None] | None = Field(
        default=None, description="Physical unit for each dimension."
    )


class Zarr3Metadata(TensorStoreModel):
    """Zarr v3 metadata specification."""

    zarr_format: Literal[3] | None = None
    node_type: Literal["array"] | None = None
    shape: list[NonNegativeInt] | None = Field(
        default=None,
        description="Array shape. Required when creating a new array "
        "if the `Schema.domain` is not otherwise specified.",
    )
    data_type: Zarr3DataType | None = Field(
        default=None,
        description="Data type specification",
        json_schema_extra=since({**DTYPE_SINCE, "r<N>": "0.1.85"}),
    )
    chunk_grid: Zarr3ChunkGrid | None = None
    chunk_key_encoding: Zarr3ChunkKeyEncoding | None = None
    fill_value: Any = Field(
        default=None, description="Fill value for uninitialized chunks"
    )
    codecs: Zarr3CodecChain | None = Field(
        default=None, description="Codec pipeline for compression and encoding"
    )
    attributes: Zarr3Attributes | None = Field(
        default=None, description="User-defined attributes"
    )
    dimension_names: list[str | None] | None = Field(
        default=None, description="Names for each dimension"
    )
    storage_transformers: list[Any] | None = Field(
        default=None,
        description="Storage transformers. tensorstore only supports the empty "
        "list, which is what zarr-python writes.",
    )

    _v: Any = field_validator("dimension_names", mode="after")(
        classmethod(_validate_labels)
    )

    @model_validator(mode="after")
    def _validate_chunk_shape_length(self) -> Self:
        """Validate that chunk_shape length matches array shape length."""
        if (
            self.shape is not None
            and self.chunk_grid is not None
            and self.chunk_grid.configuration is not None
            and self.chunk_grid.configuration.chunk_shape is not None
        ):
            shape_len = len(self.shape)
            chunk_shape_len = len(self.chunk_grid.configuration.chunk_shape)
            if shape_len != chunk_shape_len:
                raise ValueError(
                    f"chunk_shape length ({chunk_shape_len}) must match "
                    f"shape length ({shape_len})"
                )
        return self


class Zarr3Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Zarr3 driver specification for Zarr v3 format."""

    driver: Literal["zarr3"] = "zarr3"
    metadata: Zarr3Metadata | None = None
    field: str | None = Field(
        default=None,
        description="Name of the field to open (structured data types only).",
        json_schema_extra=since("0.1.85"),
    )
    open_as_void: bool | None = Field(
        default=None,
        description="Open the array as raw bytes with an extra byte dimension. "
        "Default: `false`.",
        json_schema_extra=since("0.1.85"),
    )


class Zarr3Codec(CodecBase):
    """Zarr3 codec specification."""

    driver: Literal["zarr3"] = "zarr3"
    codecs: Zarr3CodecChain | None = Field(
        default=None, description="Specifies a chain of codecs."
    )
