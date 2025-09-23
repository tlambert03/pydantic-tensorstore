"""Tests for TensorStore specification validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pydantic_tensorstore import validate_spec


def test_validate_basic_array_spec() -> None:
    """Test validation of basic array specification."""
    spec_dict = {"driver": "array", "array": [[1, 2, 3], [4, 5, 6]], "dtype": "int32"}

    # Should validate successfully
    validated = validate_spec(spec_dict)
    assert validated.driver == "array"

    # Should parse successfully
    spec = validate_spec(spec_dict)
    assert spec.driver == "array"


def test_validate_zarr_spec() -> None:
    """Test validation of Zarr specification."""
    spec_dict = {
        "driver": "zarr",
        "kvstore": {"driver": "memory"},
        "metadata": {"chunks": [64, 64], "compressor": {"id": "blosc", "cname": "lz4"}},
    }

    validated = validate_spec(spec_dict)
    assert validated.driver == "zarr"
    assert validated.kvstore

    spec = validate_spec(spec_dict)
    assert spec.driver == "zarr"


def test_validate_unknown_driver() -> None:
    """Test validation with unknown driver."""
    spec_dict = {"driver": "unknown_driver", "some_field": "value"}

    with pytest.raises(
        ValidationError, match="does not match any of the expected tags"
    ):
        validate_spec(spec_dict)


def test_validate_missing_driver() -> None:
    """Test validation with missing driver."""
    spec_dict = {"array": [[1, 2], [3, 4]], "dtype": "int32"}

    with pytest.raises(
        ValidationError, match="Unable to extract tag using discriminator"
    ):
        validate_spec(spec_dict)


def test_validate_array_missing_required_field() -> None:
    """Test validation with missing required fields for array driver."""
    spec_dict = {
        "driver": "array",
        "dtype": "int32",
        # Missing "array" field
    }

    with pytest.raises(ValidationError, match="Field required"):
        validate_spec(spec_dict)


def test_validate_zarr_missing_kvstore() -> None:
    """Test validation with missing kvstore for Zarr driver."""
    spec_dict = {
        "driver": "zarr",
        "path": "test.zarr",
        # Missing "kvstore" field
    }

    with pytest.raises(ValidationError, match="Field required"):
        validate_spec(spec_dict)


def test_validate_cross_field_consistency() -> None:
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
    validated = validate_spec(spec_dict, strict=True)
    assert validated.driver == "array"


def test_validate_already_parsed_spec() -> None:
    """Test validation of already parsed specification."""
    spec = validate_spec(
        {"driver": "array", "array": [[1, 2], [3, 4]], "dtype": "int32"}
    )

    # Should validate successfully
    validated_spec = validate_spec(spec)
    assert validated_spec.driver == "array"
    assert validated_spec is spec  # Should return the same object


def test_validate_invalid_spec_type() -> None:
    """Test validation with invalid input type."""
    with pytest.raises(
        ValidationError, match="Input should be a valid dictionary or object"
    ):
        validate_spec("not a dict or spec")


def test_validation_modes() -> None:
    """Test different validation modes."""
    spec_dict = {"driver": "array", "array": [[1, 2], [3, 4]], "dtype": "int32"}

    # Strict mode
    validated_strict = validate_spec(spec_dict, strict=True)
    assert validated_strict.driver == "array"

    # Non-strict mode
    validated_non_strict = validate_spec(spec_dict, strict=False)
    assert validated_non_strict.driver == "array"
