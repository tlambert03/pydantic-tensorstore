"""Basic usage examples for pydantic-tensorstore.

This script demonstrates the basic functionality of the pydantic-tensorstore
library for creating and validating TensorStore specifications.
"""

from __future__ import annotations

import numpy as np

from pydantic_tensorstore import (
    ArraySpec,
    N5Spec,
    Zarr2Spec,
    validate_spec,
)


def example_array_spec() -> None:
    """Example: Creating Array specifications."""
    print("=== Array Spec Examples ===")

    # Direct creation
    array_spec = ArraySpec(driver="array", array=[[1, 2, 3], [4, 5, 6]], dtype="int32")
    print(f"Array spec shape: {array_spec.get_array_shape()}")
    print(f"Array spec ndim: {array_spec.get_array_ndim()}")

    # With NumPy array
    numpy_data = np.random.randn(10, 20).astype(np.float32)
    numpy_spec = ArraySpec(driver="array", array=numpy_data, dtype="float32")
    print(f"NumPy spec shape: {numpy_spec.get_array_shape()}")


def example_zarr_spec() -> None:
    """Example: Creating Zarr specifications."""
    print("\n=== Zarr Spec Examples ===")

    # Direct creation
    zarr_spec = Zarr2Spec(
        driver="zarr",
        kvstore={"driver": "memory"},
        path="my_array.zarr",
        metadata={
            "chunks": [64, 64],
            "compressor": {"id": "blosc", "cname": "lz4"},
            "order": "C",
        },
    )
    print(f"Zarr spec: {zarr_spec}")

    # With file storage
    file_zarr_spec = Zarr2Spec(
        driver="zarr",
        kvstore={"driver": "file", "path": "/tmp/zarr_data/"},
        path="dataset.zarr",
    )
    print(f"File Zarr: {file_zarr_spec}")


def example_n5_spec() -> None:
    """Example: Creating N5 specifications."""
    print("\n=== N5 Spec Examples ===")

    n5_spec = N5Spec(
        driver="n5",
        kvstore={"driver": "file", "path": "/data/n5/"},
        path="dataset",
        metadata={
            "dimensions": [1000, 1000, 100],
            "blockSize": [64, 64, 64],
            "dataType": "uint16",
            "compression": {"type": "gzip"},
        },
    )
    print(f"N5 spec: {n5_spec}")


def example_validation_and_parsing() -> None:
    """Example: Validation and parsing from dictionaries."""
    print("\n=== Validation Examples ===")

    # Parse from dictionary
    spec_dict = {"driver": "array", "array": [[1, 2], [3, 4]], "dtype": "int32"}

    spec = validate_spec(spec_dict)
    print(f"Parsed spec driver: {spec.driver}")

    # JSON serialization
    json_str = spec.model_dump_json(indent=2)
    print("JSON representation:")
    print(json_str)

    # Parse from JSON
    parsed_spec = validate_spec(json_str)
    print(f"Parsed from JSON: {parsed_spec.driver}")


def example_discriminated_union() -> None:
    """Example: Demonstrating discriminated union behavior."""
    print("\n=== Discriminated Union Examples ===")

    # Different spec types parsed automatically based on driver
    specs = [
        {"driver": "array", "array": [[1, 2]], "dtype": "int32"},
        {"driver": "zarr", "kvstore": {"driver": "memory"}},
        {"driver": "n5", "kvstore": {"driver": "memory"}},
    ]

    for spec_dict in specs:
        spec = validate_spec(spec_dict)
        print(f"Parsed as: {type(spec).__name__} with driver '{spec.driver}'")


def main() -> None:
    """Run all examples."""
    print("Pydantic TensorStore Examples")
    print("=" * 40)

    example_array_spec()
    example_zarr_spec()
    example_n5_spec()
    example_validation_and_parsing()
    example_discriminated_union()

    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    main()
