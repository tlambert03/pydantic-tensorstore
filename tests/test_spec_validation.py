"""Tests for TensorStore specification validation."""

from __future__ import annotations

import pytest

from pydantic_tensorstore import TensorStoreSpec
from pydantic_tensorstore.validation.errors import TensorStoreValidationError
from pydantic_tensorstore.validation.validators import validate_spec, validate_spec_dict


def test_validate_basic_array_spec():
    """Test validation of basic array specification."""
    spec_dict = {"driver": "array", "array": [[1, 2, 3], [4, 5, 6]], "dtype": "int32"}

    # Should validate successfully
    validated = validate_spec_dict(spec_dict)
    assert validated["driver"] == "array"

    # Should parse successfully
    spec = validate_spec(spec_dict)
    assert spec.driver == "array"


def test_validate_zarr_spec():
    """Test validation of Zarr specification."""
    spec_dict = {
        "driver": "zarr",
        "kvstore": {"driver": "memory"},
        "metadata": {"chunks": [64, 64], "compressor": {"id": "blosc", "cname": "lz4"}},
    }

    validated = validate_spec_dict(spec_dict)
    assert validated["driver"] == "zarr"
    assert "kvstore" in validated

    spec = validate_spec(spec_dict)
    assert spec.driver == "zarr"


def test_validate_unknown_driver():
    """Test validation with unknown driver."""
    spec_dict = {"driver": "unknown_driver", "some_field": "value"}

    with pytest.raises(TensorStoreValidationError, match="Unknown driver"):
        validate_spec_dict(spec_dict)


def test_validate_missing_driver():
    """Test validation with missing driver."""
    spec_dict = {"array": [[1, 2], [3, 4]], "dtype": "int32"}

    with pytest.raises(
        TensorStoreValidationError, match="Missing required 'driver' field"
    ):
        validate_spec_dict(spec_dict)


def test_validate_array_missing_required_field():
    """Test validation with missing required fields for array driver."""
    spec_dict = {
        "driver": "array",
        "dtype": "int32",
        # Missing "array" field
    }

    with pytest.raises(
        TensorStoreValidationError, match="Array driver requires 'array' field"
    ):
        validate_spec_dict(spec_dict)


def test_validate_zarr_missing_kvstore():
    """Test validation with missing kvstore for Zarr driver."""
    spec_dict = {
        "driver": "zarr",
        "path": "test.zarr",
        # Missing "kvstore" field
    }

    with pytest.raises(
        TensorStoreValidationError, match="Zarr driver requires 'kvstore' field"
    ):
        validate_spec_dict(spec_dict)


def test_validate_cross_field_consistency():
    """Test cross-field validation."""
    # This would test more complex validation scenarios
    # For now, just test that basic validation passes
    spec_dict = {
        "driver": "array",
        "array": [[1, 2], [3, 4]],
        "dtype": "int32",
        "schema": {"rank": 2},
    }

    # Should validate successfully
    validated = validate_spec_dict(spec_dict, strict=True)
    assert validated["driver"] == "array"


def test_validate_already_parsed_spec():
    """Test validation of already parsed specification."""
    spec = TensorStoreSpec.model_validate(
        {"driver": "array", "array": [[1, 2], [3, 4]], "dtype": "int32"}
    )

    # Should validate successfully
    validated_spec = validate_spec(spec)
    assert validated_spec.driver == "array"
    assert validated_spec is spec  # Should return the same object


def test_validate_invalid_spec_type():
    """Test validation with invalid input type."""
    with pytest.raises(TensorStoreValidationError, match="Invalid spec type"):
        validate_spec("not a dict or spec")


def test_validation_modes():
    """Test different validation modes."""
    spec_dict = {"driver": "array", "array": [[1, 2], [3, 4]], "dtype": "int32"}

    # Strict mode
    validated_strict = validate_spec_dict(spec_dict, strict=True)
    assert validated_strict["driver"] == "array"

    # Non-strict mode
    validated_non_strict = validate_spec_dict(spec_dict, strict=False)
    assert validated_non_strict["driver"] == "array"
