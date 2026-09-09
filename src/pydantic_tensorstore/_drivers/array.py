"""Array driver specification for in-memory arrays."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, GetCoreSchemaHandler, model_validator
from pydantic_core import core_schema

from pydantic_tensorstore._core.spec import BaseSpec
from pydantic_tensorstore._types import ContextResource, DataType


class ArrayValidator:
    """Pydantic-compatible numpy array (validated from nested lists)."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _serialize(val: np.ndarray) -> list:
            return val.tolist()  # type: ignore[no-any-return]

        def _validate_array(val: Any) -> np.ndarray:
            return val if isinstance(val, np.ndarray) else np.asarray(val)

        ser_schema = core_schema.plain_serializer_function_ser_schema(
            _serialize, return_schema=core_schema.list_schema()
        )
        return core_schema.no_info_before_validator_function(
            _validate_array,
            core_schema.any_schema(),
            serialization=ser_schema,
        )


class ArraySpec(BaseSpec):
    """Array driver specification for in-memory arrays."""

    driver: Literal["array"] = "array"
    dtype: DataType | None = Field(
        default=None,
        description="Data type. Required, except when nested as the `base` of an "
        "adapter driver (tensorstore then hoists it to the adapter).",
    )
    array: Annotated[np.ndarray, ArrayValidator] = Field(
        description="Nested array data or NumPy array",
    )
    data_copy_concurrency: ContextResource | None = Field(
        default=None,
        description='Concurrency for copying. Default: `"data_copy_concurrency"`.',
    )

    @model_validator(mode="after")
    def _validate_array(self) -> Self:
        """Cast to the declared dtype when numpy knows it; check rank."""
        try:
            np_dtype = np.dtype(str(self.dtype)) if self.dtype is not None else None
        except TypeError:
            np_dtype = None
        if np_dtype is not None and self.array.dtype != np_dtype:
            object.__setattr__(self, "array", self.array.astype(np_dtype))
        if self.rank is not None and self.rank != self.array.ndim:
            raise ValueError(
                f"Specified rank ({self.rank}) does not match array dimensions "
                f"({self.array.ndim})"
            )
        return self
