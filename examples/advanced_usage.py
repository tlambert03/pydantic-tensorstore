"""Advanced usage examples for pydantic-tensorstore.

Demonstrates more complex scenarios like custom validation,
spec merging, and integration patterns.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_tensorstore import TensorStoreSpec
from pydantic_tensorstore.validation.validators import validate_spec
from pydantic_tensorstore.utils.conversion import merge_specs, compare_specs, normalize_spec
from pydantic_tensorstore.utils.introspection import get_driver_capabilities, get_compatible_drivers
from pydantic_tensorstore.validation.errors import TensorStoreValidationError


def example_custom_validation():
    """Example: Custom validation logic."""
    print("=== Custom Validation Examples ===")

    # Valid spec
    valid_spec = {
        "driver": "zarr",
        "kvstore": {"driver": "memory"},
        "metadata": {"chunks": [64, 64]}
    }

    try:
        spec = validate_spec(valid_spec, strict=True)
        print(f"✅ Valid spec: {spec.driver}")
    except TensorStoreValidationError as e:
        print(f"❌ Validation error: {e}")

    # Invalid spec (missing kvstore)
    invalid_spec = {
        "driver": "zarr",
        "path": "test.zarr"
    }

    try:
        spec = validate_spec(invalid_spec, strict=True)
        print(f"✅ Parsed: {spec.driver}")
    except TensorStoreValidationError as e:
        print(f"❌ Expected validation error: {e}")


def example_spec_merging():
    """Example: Merging specifications."""
    print("\n=== Spec Merging Examples ===")

    base_spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": "/data/"},
        "metadata": {"chunks": [64, 64]}
    }

    override_spec = {
        "driver": "zarr",
        "path": "my_array.zarr",
        "metadata": {"compressor": {"id": "blosc"}}
    }

    merged = merge_specs(base_spec, override_spec)
    print("Merged spec:")
    print(json.dumps(merged.model_dump(exclude_unset=True), indent=2))


def example_spec_comparison():
    """Example: Comparing specifications."""
    print("\n=== Spec Comparison Examples ===")

    spec1 = {
        "driver": "array",
        "array": [[1, 2], [3, 4]],
        "dtype": "int32"
    }

    spec2 = {
        "driver": "array",
        "array": [[1, 2], [3, 4]],
        "dtype": "int32",
        "context": {"cache_pool": {}}  # Extra field
    }

    # Compare with all fields
    are_equal = compare_specs(spec1, spec2)
    print(f"Specs equal (all fields): {are_equal}")

    # Compare ignoring context
    are_equal_ignoring_context = compare_specs(spec1, spec2, ignore_fields=["context"])
    print(f"Specs equal (ignoring context): {are_equal_ignoring_context}")


def example_driver_capabilities():
    """Example: Exploring driver capabilities."""
    print("\n=== Driver Capabilities Examples ===")

    drivers = ["array", "zarr", "n5"]

    for driver in drivers:
        caps = get_driver_capabilities(driver)
        print(f"\n{driver.upper()} Driver:")
        print(f"  Supports kvstore: {caps['supports_kvstore']}")
        print(f"  Supports compression: {caps['supports_compression']}")
        print(f"  Supported dtypes: {len(caps['supported_dtypes'])} types")
        print(f"  Description: {caps['description']}")


def example_compatible_drivers():
    """Example: Finding compatible drivers."""
    print("\n=== Compatible Drivers Examples ===")

    # Find drivers that support float32 and compression
    requirements = {
        "dtype": "float32",
        "needs_compression": True
    }

    compatible = get_compatible_drivers(requirements)
    print(f"Drivers supporting float32 + compression: {compatible}")

    # Find drivers that need kvstore
    requirements = {
        "needs_kvstore": True
    }

    compatible = get_compatible_drivers(requirements)
    print(f"Drivers using kvstore: {compatible}")


def example_spec_normalization():
    """Example: Normalizing specifications."""
    print("\n=== Spec Normalization Examples ===")

    # Raw dictionary with extra/default fields
    raw_spec = {
        "driver": "array",
        "array": [[1, 2], [3, 4]],
        "dtype": "int32",
        "context": None,  # Default value
        "extra_field": "should_be_removed"  # Invalid field
    }

    try:
        normalized = normalize_spec(raw_spec)
        print("Normalized spec:")
        print(json.dumps(normalized, indent=2))
    except Exception as e:
        print(f"Normalization error: {e}")


def example_conditional_spec_building():
    """Example: Building specs conditionally."""
    print("\n=== Conditional Spec Building ===")

    def build_storage_spec(
        storage_type: str,
        path: str,
        use_compression: bool = True
    ) -> dict[str, Any]:
        """Build a storage spec based on requirements."""
        if storage_type == "memory":
            spec = {
                "driver": "zarr",
                "kvstore": {"driver": "memory"},
                "path": path
            }
        elif storage_type == "file":
            spec = {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": "/data/"},
                "path": path
            }
        elif storage_type == "n5":
            spec = {
                "driver": "n5",
                "kvstore": {"driver": "file", "path": "/data/n5/"},
                "path": path
            }
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

        # Add compression if requested
        if use_compression and spec["driver"] in ["zarr", "n5"]:
            if "metadata" not in spec:
                spec["metadata"] = {}

            if spec["driver"] == "zarr":
                spec["metadata"]["compressor"] = {"id": "blosc", "cname": "lz4"}
            elif spec["driver"] == "n5":
                spec["metadata"]["compression"] = {"type": "gzip"}

        return spec

    # Build different storage specs
    storage_types = ["memory", "file", "n5"]

    for storage_type in storage_types:
        spec_dict = build_storage_spec(storage_type, f"test_{storage_type}")
        spec = TensorStoreSpec.model_validate(spec_dict)
        print(f"{storage_type}: {spec.driver} driver")


def example_error_handling():
    """Example: Proper error handling."""
    print("\n=== Error Handling Examples ===")

    invalid_specs = [
        {"driver": "unknown_driver"},
        {"array": [[1, 2]], "dtype": "int32"},  # Missing driver
        {"driver": "zarr"},  # Missing kvstore
        {"driver": "array", "array": [], "dtype": "int32"},  # Empty array
    ]

    for i, spec_dict in enumerate(invalid_specs, 1):
        try:
            spec = validate_spec(spec_dict)
            print(f"Spec {i}: ✅ Valid")
        except TensorStoreValidationError as e:
            print(f"Spec {i}: ❌ {e}")
        except Exception as e:
            print(f"Spec {i}: ❌ Unexpected error: {e}")


def main():
    """Run all advanced examples."""
    print("Advanced Pydantic TensorStore Examples")
    print("=" * 45)

    example_custom_validation()
    example_spec_merging()
    example_spec_comparison()
    example_driver_capabilities()
    example_compatible_drivers()
    example_spec_normalization()
    example_conditional_spec_building()
    example_error_handling()

    print("\n✅ All advanced examples completed!")


if __name__ == "__main__":
    main()