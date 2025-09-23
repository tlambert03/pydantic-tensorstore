"""Common types and enums used throughout TensorStore specifications."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, StringConstraints

__all__ = [
    "ChunkShape",
    "ContextResource",
    "DataType",
    "DimensionIndex",
    "DomainShape",
    "Index",
    "OpenMode",
    "ReadWriteMode",
    "Shape",
]

# Basic index types
DimensionIndex: TypeAlias = int
Index: TypeAlias = int
Shape: TypeAlias = list[int]
ChunkShape: TypeAlias = list[int | None]
DomainShape: TypeAlias = list[int | Literal["*"]]


class DataType(str, Enum):
    """TensorStore data types.

    Based on the supported data types from TensorStore's C++ implementation.
    Supports bool, signed/unsigned integers, floating point, and complex types.
    """

    # Boolean
    BOOL = "bool"

    # Signed integers
    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"

    # Unsigned integers
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"

    # Floating point
    FLOAT8_E3M4 = "float8_e3m4"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E4M3FNUZ = "float8_e4m3fnuz"
    FLOAT8_E4M3B11FNUZ = "float8_e4m3b11fnuz"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT8_E5M2FNUZ = "float8_e5m2fnuz"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    # Complex
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"

    # String types
    STRING = "string"
    USTRING = "ustring"

    # JSON
    JSON = "json"


class OpenMode(str, Enum):
    """TensorStore open modes.

    Controls how TensorStore opens or creates datasets.
    """

    OPEN = "open"
    CREATE = "create"
    DELETE_EXISTING = "delete_existing"
    ASSUME_METADATA = "assume_metadata"
    ASSUME_CACHED_METADATA = "assume_cached_metadata"


class ReadWriteMode(str, Enum):
    """Read/write access modes."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


ContextResource: TypeAlias = dict | bool | float | int | str | None
"""Specifies a context resource of a particular <resource-type>."""


class Unit(BaseModel):
    """Physical unit specification."""

    multiplier: float = Field(default=1.0, description="Unit multiplier")
    base_unit: str = Field(default="", description="Base unit (e.g., 'm', 's', 'nm')")

    def __str__(self) -> str:
        """Return string representation of the unit."""
        if self.base_unit == "":
            return str(self.multiplier) if self.multiplier != 1.0 else ""
        if self.multiplier == 1.0:
            return self.base_unit
        return f"{self.multiplier}{self.base_unit}"


# String constraints for identifiers
DriverName = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")]
ContextResourceName = Annotated[str, StringConstraints(min_length=1)]

# Common JSON-like types
JsonValue: TypeAlias = str | int | float | bool | None | dict[str, Any] | list[Any]
JsonObject: TypeAlias = dict[str, JsonValue]
