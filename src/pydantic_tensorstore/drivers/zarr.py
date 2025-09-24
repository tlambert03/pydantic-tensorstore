"""Zarr driver specification for Zarr v2 format."""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from pydantic_tensorstore._types import JsonObject
from pydantic_tensorstore.core.codec import CodecBase
from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec


class ZarrMetadata(BaseModel):
    """Zarr metadata specification.

    Controls Zarr-specific format options like compression,
    chunk shapes, and array metadata.
    """

    model_config: ClassVar = {"extra": "allow"}

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


class Zarr2Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """Zarr driver specification for Zarr v2 format."""

    driver: Literal["zarr"] = "zarr"

    metadata: ZarrMetadata | None = Field(
        default=None,
        description="Zarr metadata specification",
    )


class Zarr2Codec(CodecBase):
    """Zarr2 codec specification."""

    driver: Literal["zarr"] = "zarr"
