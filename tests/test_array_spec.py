"""Tests for Array driver specification."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from pydantic_tensorstore import ArraySpec, DataType


def test_array_spec_basic() -> None:
    """Test basic array spec creation."""
    spec = ArraySpec(driver="array", array=[[1, 2, 3], [4, 5, 6]], dtype="int32")

    assert spec.driver == "array"
    assert spec.array == [[1, 2, 3], [4, 5, 6]]
    assert spec.dtype == DataType.INT32
    assert spec.get_array_shape() == [2, 3]
    assert spec.get_array_ndim() == 2
    assert spec.model_dump_json()


def test_array_spec_numpy() -> None:
    """Test array spec with NumPy array."""
    arr = np.random.randn(10, 20).astype(np.float32)
    spec = ArraySpec(driver="array", array=arr, dtype="float32")

    assert spec.get_array_shape() == [10, 20]
    np.testing.assert_array_equal(spec.to_numpy(), arr)


def test_array_spec_validation_errors() -> None:
    """Test validation errors for array spec."""
    # Empty array
    with pytest.raises(ValidationError):
        ArraySpec(driver="array", array=[], dtype="int32")

    # Inconsistent nested list
    with pytest.raises(ValidationError):
        ArraySpec(
            driver="array",
            array=[[1, 2], [3, 4, 5]],  # Different lengths
            dtype="int32",
        )

    # Invalid dtype
    with pytest.raises(ValidationError):
        ArraySpec(driver="array", array=[[1, 2], [3, 4]], dtype="invalid_type")


def test_array_spec_dtype_conversion() -> None:
    """Test data type conversion."""
    spec = ArraySpec(driver="array", array=[[1.0, 2.0], [3.0, 4.0]], dtype="float64")

    arr = spec.to_numpy()
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, [[1.0, 2.0], [3.0, 4.0]])


def test_array_spec_3d() -> None:
    """Test 3D array specification."""
    array_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    spec = ArraySpec(driver="array", array=array_3d, dtype="int32")

    assert spec.get_array_shape() == [2, 2, 2]
    assert spec.get_array_ndim() == 3
