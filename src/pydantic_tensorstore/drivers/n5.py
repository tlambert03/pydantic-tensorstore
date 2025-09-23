"""N5 driver specification for N5 format."""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from pydantic_tensorstore.core.spec import ChunkedTensorStoreKvStoreAdapterSpec


class N5Metadata(BaseModel):
    """N5 metadata specification.

    N5 is a chunked array storage format similar to Zarr,
    developed by the Saalfeld lab at HHMI Janelia.
    """

    model_config: ClassVar = {"extra": "allow"}

    dimensions: list[int] = Field(
        description="Array dimensions",
    )

    blockSize: list[int] = Field(
        description="Block (chunk) size for each dimension",
    )

    dataType: str = Field(
        description="N5 data type specification",
    )

    compression: dict[str, Any] = Field(
        default_factory=lambda: {"type": "raw"},
        description="Compression configuration",
    )

    axes: list[str] | None = Field(
        default=None,
        description="Axis labels for each dimension",
    )

    units: list[str] | None = Field(
        default=None,
        description="Physical units for each dimension",
    )

    resolution: list[float] | None = Field(
        default=None,
        description="Resolution (pixel size) for each dimension",
    )

    offset: list[float] | None = Field(
        default=None,
        description="Offset for each dimension",
    )

    @field_validator("dimensions", "blockSize", mode="before")
    @classmethod
    def validate_int_list(cls, v: Any) -> list[int]:
        """Validate integer list fields."""
        if not isinstance(v, list):
            raise ValueError("Must be a list of integers")

        for i, val in enumerate(v):
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"Element {i} must be a positive integer")

        return v

    @field_validator("dataType", mode="before")
    @classmethod
    def validate_data_type(cls, v: Any) -> str:
        """Validate N5 data type specification."""
        if not isinstance(v, str):
            raise ValueError("dataType must be a string")

        # N5 uses specific type names
        valid_types = {
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        }

        if v not in valid_types:
            raise ValueError(f"Invalid N5 dataType '{v}'. Valid types: {valid_types}")

        return v


class N5Spec(ChunkedTensorStoreKvStoreAdapterSpec):
    """N5 driver specification for N5 format."""

    driver: Literal["n5"] = "n5"

    metadata: N5Metadata | None = Field(
        default=None,
        description="N5 metadata specification",
    )
