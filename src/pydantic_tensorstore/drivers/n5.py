"""N5 driver specification for N5 format."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator

from pydantic_tensorstore._types import DataType, JsonObject
from pydantic_tensorstore.core.spec import BaseDriverSpec
from pydantic_tensorstore.kvstore import KvStoreSpec  # noqa: TC001


class N5Metadata(BaseDriverSpec):
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


class N5Spec(BaseDriverSpec):
    """N5 driver specification for N5 format.

    N5 (Not HDF5) is a chunked array storage format that's popular
    in scientific computing, especially for large-scale image data.

    Attributes
    ----------
        driver: Must be "n5"
        kvstore: Key-value store for data storage
        path: Path within the kvstore for this array
        metadata: N5-specific metadata and options

    Example:
        >>> spec = N5Spec(
        ...     driver="n5",
        ...     kvstore={"driver": "file", "path": "/data/n5/"},
        ...     path="dataset",
        ...     metadata={
        ...         "dimensions": [1000, 1000, 100],
        ...         "blockSize": [64, 64, 64],
        ...         "dataType": "uint16",
        ...         "compression": {"type": "gzip"},
        ...     },
        ... )
    """

    model_config: ClassVar = {"extra": "forbid"}

    driver: Literal["n5"] = Field(
        default="n5",
        description="N5 driver identifier",
    )

    kvstore: KvStoreSpec | JsonObject = Field(
        description="Key-value store for data storage",
    )

    path: str = Field(
        default="",
        description="Path within the kvstore for this dataset",
    )

    metadata: N5Metadata | JsonObject | None = Field(
        default=None,
        description="N5 metadata specification",
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

    def tensorstore_dtype_to_n5(self, tensorstore_dtype: DataType) -> str:
        """Convert TensorStore DataType to N5 dataType string."""
        mapping = {
            DataType.UINT8: "uint8",
            DataType.UINT16: "uint16",
            DataType.UINT32: "uint32",
            DataType.UINT64: "uint64",
            DataType.INT8: "int8",
            DataType.INT16: "int16",
            DataType.INT32: "int32",
            DataType.INT64: "int64",
            DataType.FLOAT32: "float32",
            DataType.FLOAT64: "float64",
        }

        if tensorstore_dtype not in mapping:
            raise ValueError(f"DataType {tensorstore_dtype} not supported by N5")

        return mapping[tensorstore_dtype]

    def n5_dtype_to_tensorstore(self, n5_dtype: str) -> DataType:
        """Convert N5 dataType string to TensorStore DataType."""
        mapping = {
            "uint8": DataType.UINT8,
            "uint16": DataType.UINT16,
            "uint32": DataType.UINT32,
            "uint64": DataType.UINT64,
            "int8": DataType.INT8,
            "int16": DataType.INT16,
            "int32": DataType.INT32,
            "int64": DataType.INT64,
            "float32": DataType.FLOAT32,
            "float64": DataType.FLOAT64,
        }

        if n5_dtype not in mapping:
            raise ValueError(f"N5 dataType '{n5_dtype}' not supported")

        return mapping[n5_dtype]
