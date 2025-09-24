"""Common types and enums used throughout TensorStore specifications."""

import re
from collections.abc import Sequence
from contextlib import suppress
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    StringConstraints,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema

__all__ = [
    "ChunkShape",
    "ContextResource",
    "DataType",
    "DomainShape",
    "OpenMode",
    "ReadWriteMode",
    "Shape",
]

# Basic index types
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

    def __str__(self) -> str:
        return self.value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        members = list(cls.__members__.values())
        schema = core_schema.enum_schema(cls, members=members)

        def _cast_to_dtype(v: Any) -> Any:
            if isinstance(v, str):
                with suppress(ValueError):
                    return DataType(v)
                with suppress(ValueError, TypeError):
                    return DataType(np.dtype(v).name)
                raise ValueError(
                    f"Invalid string data type: {v}.  Must be one of {members}"
                )
            return v

        return core_schema.no_info_before_validator_function(
            function=_cast_to_dtype,
            schema=schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                function=str,
                return_schema=core_schema.str_schema(),
            ),
        )


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
    base_unit: str = Field(
        default="",
        description=(
            "A base_unit, represented as a string. An empty string may be used to "
            "indicate a dimensionless quantity. In general, TensorStore does not "
            "interpret the base unit string; some drivers impose additional "
            "constraints on the base unit, while other drivers may store the specified "
            "unit directly. It is recommended to follow the udunits2 syntax unless "
            "there is a specific need to deviate."
        ),
    )

    def __str__(self) -> str:
        """Return string representation of the unit."""
        if self.base_unit == "":
            return str(self.multiplier) if self.multiplier != 1.0 else ""
        if self.multiplier == 1.0:
            return self.base_unit
        return f"{self.multiplier}{self.base_unit}"

    @model_validator(mode="before")
    def _validate_unit(cls, v: Any) -> Any:
        """Three JSON formats are supported.

        - The canonical format, as a two-element [multiplier, base_unit] array. This
          format is always used by TensorStore when returning the JSON representation of
          a unit.

        - A single string. If the string contains a leading number, it is parsed as the
          multiplier and the remaining portion, after stripping leading and trailing
          whitespace, is used as the base_unit. If there is no leading number, the
          multiplier is 1 and the entire string, after stripping leading and trailing
          whitespace, is used as the base_unit.

        - A single number, to indicate a dimension-less unit with the specified
          multiplier.
        """
        if isinstance(v, (float, int)):
            return {"multiplier": float(v), "base_unit": ""}
        if isinstance(v, str):
            # regex to match a leading float (including scientific notation)

            match = re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", v)
            if match:
                multiplier_str = match.group(0)
                base_unit = v[match.end() :].strip()
                return {"multiplier": float(multiplier_str), "base_unit": base_unit}
            else:
                return {"multiplier": 1.0, "base_unit": v.strip()}
        elif isinstance(v, Sequence):
            if len(v) != 2:
                raise ValueError("Unit array must have exactly two elements")
            return {"multiplier": float(v[0]), "base_unit": str(v[1])}
        return v


# String constraints for identifiers
DriverName = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")]
ContextResourceName = Annotated[str, StringConstraints(min_length=1)]
