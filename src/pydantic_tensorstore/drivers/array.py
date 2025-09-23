"""Array driver specification for in-memory arrays."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator

from pydantic_tensorstore._types import DataType, JsonObject
from pydantic_tensorstore.core.spec import BaseDriverSpec


class ArraySpec(BaseDriverSpec):
    """Array driver specification for in-memory arrays.

    Creates a TensorStore backed by an in-memory NumPy-like array.
    Useful for testing and small datasets that fit in memory.

    Attributes
    ----------
        driver: Must be "array"
        array: Nested list or NumPy array containing the data
        dtype: Data type of the array elements
        data_copy_concurrency: Concurrency resource for data operations

    Example:
        >>> spec = ArraySpec(
        ...     driver="array", array=[[1, 2, 3], [4, 5, 6]], dtype="int32"
        ... )
        >>> # With NumPy array
        >>> import numpy as np
        >>> spec = ArraySpec(
        ...     driver="array", array=np.random.randn(10, 20), dtype="float64"
        ... )
    """

    model_config: ClassVar = {"extra": "forbid", "arbitrary_types_allowed": True}

    driver: Literal["array"] = Field(
        default="array",
        description="Array driver identifier",
    )

    array: list[Any] | np.ndarray = Field(
        description="Nested array data or NumPy array",
    )

    dtype: DataType = Field(
        description="Data type of array elements",
    )

    data_copy_concurrency: str | JsonObject | None = Field(
        default="data_copy_concurrency",
        description="Data copy concurrency resource",
    )

    @field_validator("array", mode="before")
    @classmethod
    def validate_array_data(cls, v: Any) -> Any:
        """Validate array data structure."""
        if isinstance(v, np.ndarray):
            return v

        if isinstance(v, list):
            # Validate that it's a rectangular nested list
            if not v:
                raise ValueError("Array cannot be empty")

            # Check that all rows have the same length (for 2D+)
            def check_rectangular(
                arr: list[Any], depth: int = 0
            ) -> tuple[list[int], int]:
                """Check if nested list is rectangular and return shape."""
                if not isinstance(arr, list):
                    return [], depth

                if not arr:
                    raise ValueError(f"Empty subarray at depth {depth}")

                # Check first element to determine if this is the deepest level
                first_elem = arr[0]
                if isinstance(first_elem, list):
                    # Recursive case - check all sublists have same shape
                    first_shape, first_depth = check_rectangular(first_elem, depth + 1)
                    shape = [len(arr), *first_shape]

                    for i, elem in enumerate(arr[1:], 1):
                        if not isinstance(elem, list):
                            raise ValueError(
                                f"Inconsistent nesting: element {i} at depth {depth} "
                                f"is not a list while element 0 is"
                            )
                        elem_shape, elem_depth = check_rectangular(elem, depth + 1)
                        if elem_shape != first_shape or elem_depth != first_depth:
                            raise ValueError(
                                f"Inconsistent shape: element {i} at depth {depth} "
                                f"has shape {elem_shape} vs expected {first_shape}"
                            )

                    return shape, first_depth
                else:
                    # Base case - this is a 1D array of scalars
                    return [len(arr)], depth + 1

            _shape, _depth = check_rectangular(v)
            return v

        try:
            # Try to convert to NumPy array for validation
            arr = np.asarray(v)
            return arr.tolist()  # Convert back to list for JSON serialization
        except Exception as e:
            raise ValueError(f"Invalid array data: {e}") from e

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype_compatibility(cls, v: Any) -> Any:
        """Validate data type."""
        if isinstance(v, str):
            try:
                return DataType(v)
            except ValueError:
                valid_types = [dt.value for dt in DataType]
                raise ValueError(
                    f"Invalid dtype '{v}'. Valid types: {valid_types}"
                ) from None
        return v

    def get_array_shape(self) -> list[int]:
        """Get the shape of the array data."""
        if isinstance(self.array, np.ndarray):
            return list(self.array.shape)

        # Calculate shape from nested list
        def get_shape(arr: list[Any]) -> list[int]:
            if not isinstance(arr, list) or not arr:
                return []
            shape = [len(arr)]
            if isinstance(arr[0], list):
                shape.extend(get_shape(arr[0]))
            return shape

        return get_shape(self.array)

    def get_array_ndim(self) -> int:
        """Get the number of dimensions."""
        return len(self.get_array_shape())

    def to_numpy(self) -> np.ndarray:
        """Convert the array data to NumPy array."""
        if isinstance(self.array, np.ndarray):
            return self.array.copy()

        # Convert from nested list
        arr = np.asarray(self.array)

        # Convert dtype if needed
        if self.dtype == DataType.BOOL:
            return arr.astype(bool)
        elif self.dtype == DataType.INT8:
            return arr.astype(np.int8)
        elif self.dtype == DataType.INT16:
            return arr.astype(np.int16)
        elif self.dtype == DataType.INT32:
            return arr.astype(np.int32)
        elif self.dtype == DataType.INT64:
            return arr.astype(np.int64)
        elif self.dtype == DataType.UINT8:
            return arr.astype(np.uint8)
        elif self.dtype == DataType.UINT16:
            return arr.astype(np.uint16)
        elif self.dtype == DataType.UINT32:
            return arr.astype(np.uint32)
        elif self.dtype == DataType.UINT64:
            return arr.astype(np.uint64)
        elif self.dtype == DataType.FLOAT16:
            return arr.astype(np.float16)
        elif self.dtype == DataType.FLOAT32:
            return arr.astype(np.float32)
        elif self.dtype == DataType.FLOAT64:
            return arr.astype(np.float64)
        elif self.dtype == DataType.COMPLEX64:
            return arr.astype(np.complex64)
        elif self.dtype == DataType.COMPLEX128:
            return arr.astype(np.complex128)
        else:
            # For other types, return as-is and let NumPy handle it
            return arr
