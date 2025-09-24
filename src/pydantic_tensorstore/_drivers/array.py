"""Array driver specification for in-memory arrays."""

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, GetCoreSchemaHandler, model_validator
from pydantic_core import core_schema
from typing_extensions import Self

from pydantic_tensorstore._core.spec import BaseSpec
from pydantic_tensorstore._types import ContextResource, DataType


class ArrayValidator:
    """Pydantic-compatible 4x4 numpy array."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _serialize(val: np.ndarray) -> list:
            return val.tolist()  # type: ignore[no-any-return]

        def _validate_array(val: Any) -> np.ndarray:
            if not isinstance(val, np.ndarray):
                val = np.asarray(val, dtype=float)
            return val

        ser_schema = core_schema.plain_serializer_function_ser_schema(
            _serialize, return_schema=core_schema.list_schema()
        )

        return core_schema.no_info_before_validator_function(
            _validate_array,
            core_schema.any_schema(),
            serialization=ser_schema,
        )


class ArraySpec(BaseSpec):
    """Array driver specification for in-memory arrays.

    Creates a TensorStore backed by an in-memory NumPy-like array.
    Useful for testing and small datasets that fit in memory.
    """

    driver: Literal["array"] = "array"

    dtype: DataType  # pyright: ignore

    array: Annotated[np.ndarray, ArrayValidator] = Field(
        description="Nested array data or NumPy array",
    )

    data_copy_concurrency: ContextResource = "data_copy_concurrency"

    @model_validator(mode="after")
    def _validate_array_rank_consistency(self) -> Self:
        """Validate that array dimensions match specified rank if provided."""
        if self.rank is not None and self.rank != self.array.ndim:
            raise ValueError(
                f"Specified rank ({self.rank}) does not match array dimensions "
                f"({self.array.ndim})"
            )

        return self
