"""Introspection utilities for TensorStore specifications."""

from __future__ import annotations

from typing import Any

from pydantic_tensorstore.core.spec import SpecValidator, TensorStoreSpec
from pydantic_tensorstore.types.common import DataType


def get_spec_info(spec: TensorStoreSpec) -> dict[str, Any]:
    """Get comprehensive information about a TensorStore specification.

    Parameters
    ----------
    spec : TensorStoreSpec
        The specification to analyze

    Returns
    -------
    dict
        Dictionary containing spec information including:
        - driver: Driver name
        - driver_kind: Kind of driver (tensorstore, kvstore)
        - dtype: Data type if available
        - shape: Shape if available
        - rank: Rank if available
        - has_kvstore: Whether spec uses a kvstore
        - storage_backend: Type of storage backend
        - compression: Compression info if available
    """
    info = {
        "driver": spec.driver,
        "driver_kind": spec.get_driver_kind(),
        "dtype": None,
        "shape": None,
        "rank": None,
        "has_kvstore": False,
        "storage_backend": None,
        "compression": None,
        "metadata": {},
    }

    # Extract schema information
    if hasattr(spec, "schema") and spec.schema:
        schema = spec.schema
        if isinstance(schema, dict):
            info["dtype"] = schema.get("dtype")
            domain = schema.get("domain")
            if domain and isinstance(domain, dict):
                info["shape"] = domain.get("shape")
                if info["shape"]:
                    info["rank"] = len(info["shape"])
        else:
            # Schema object
            info["dtype"] = str(schema.dtype) if schema.dtype else None
            if schema.domain:
                info["shape"] = schema.domain.shape
                info["rank"] = schema.domain.rank

    # Extract kvstore information
    if hasattr(spec, "kvstore") and spec.kvstore:
        info["has_kvstore"] = True
        kvstore = spec.kvstore
        if isinstance(kvstore, dict):
            info["storage_backend"] = kvstore.get("driver")
        else:
            info["storage_backend"] = kvstore.driver

    # Driver-specific information
    if spec.driver == "array":
        if hasattr(spec, "array"):
            array_data = spec.array
            if hasattr(array_data, "shape"):  # NumPy array
                info["shape"] = list(array_data.shape)
                info["rank"] = len(array_data.shape)
            elif isinstance(array_data, list):
                # Calculate shape from nested list
                shape = _get_nested_list_shape(array_data)
                info["shape"] = shape
                info["rank"] = len(shape)

    elif spec.driver in ["zarr", "zarr3", "n5"]:
        if hasattr(spec, "metadata") and spec.metadata:
            metadata = spec.metadata
            if isinstance(metadata, dict):
                info["metadata"] = metadata.copy()

                # Extract compression info
                if "compressor" in metadata:
                    info["compression"] = metadata["compressor"]
                elif "compression" in metadata:
                    info["compression"] = metadata["compression"]

                # Extract shape info for specific formats
                if spec.driver == "zarr3" and "shape" in metadata:
                    info["shape"] = metadata["shape"]
                    info["rank"] = len(metadata["shape"])
                elif spec.driver == "n5" and "dimensions" in metadata:
                    info["shape"] = metadata["dimensions"]
                    info["rank"] = len(metadata["dimensions"])

    return info


def get_driver_capabilities(driver: str) -> dict[str, Any]:
    """Get information about a driver's capabilities.

    Parameters
    ----------
    driver : str
        Driver name

    Returns
    -------
    dict
        Dictionary containing driver capabilities including:
        - supported_dtypes: List of supported data types
        - supports_kvstore: Whether driver uses kvstore
        - supports_compression: Whether driver supports compression
        - supports_chunks: Whether driver supports chunking
        - metadata_format: Format of metadata (if applicable)
    """
    capabilities = {
        "supported_dtypes": [],
        "supports_kvstore": False,
        "supports_compression": False,
        "supports_chunks": False,
        "metadata_format": None,
        "description": "",
    }

    if driver == "array":
        capabilities.update(
            {
                "supported_dtypes": [dt.value for dt in DataType],
                "supports_kvstore": False,
                "supports_compression": False,
                "supports_chunks": False,
                "description": "In-memory array driver for testing and small datasets",
            }
        )

    elif driver == "zarr":
        capabilities.update(
            {
                "supported_dtypes": [
                    "bool",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "float16",
                    "float32",
                    "float64",
                    "complex64",
                    "complex128",
                ],
                "supports_kvstore": True,
                "supports_compression": True,
                "supports_chunks": True,
                "metadata_format": "zarr_v2",
                "description": "Zarr v2 format driver for chunked, compressed arrays",
            }
        )

    elif driver == "zarr3":
        capabilities.update(
            {
                "supported_dtypes": [
                    "bool",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "float16",
                    "float32",
                    "float64",
                    "complex64",
                    "complex128",
                ],
                "supports_kvstore": True,
                "supports_compression": True,
                "supports_chunks": True,
                "metadata_format": "zarr_v3",
                "description": "Zarr v3 format driver with enhanced features",
            }
        )

    elif driver == "n5":
        capabilities.update(
            {
                "supported_dtypes": [
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "float32",
                    "float64",
                ],
                "supports_kvstore": True,
                "supports_compression": True,
                "supports_chunks": True,
                "metadata_format": "n5",
                "description": "N5 format driver for scientific computing",
            }
        )

    elif driver == "neuroglancer_precomputed":
        capabilities.update(
            {
                "supported_dtypes": ["uint8", "uint16", "uint32", "uint64", "float32"],
                "supports_kvstore": True,
                "supports_compression": True,
                "supports_chunks": True,
                "metadata_format": "neuroglancer",
                "description": "Neuroglancer Precomputed format for visualization",
            }
        )

    return capabilities


def list_registered_drivers() -> list[str]:
    """List all registered drivers.

    Returns
    -------
    list of str
        List of registered driver names
    """
    return SpecValidator.get_registered_drivers()


def validate_driver_exists(driver: str) -> bool:
    """Check if a driver is registered.

    Parameters
    ----------
    driver : str
        Driver name to check

    Returns
    -------
    bool
        True if driver is registered
    """
    return SpecValidator.is_driver_registered(driver)


def get_compatible_drivers(requirements: dict[str, Any]) -> list[str]:
    """Get drivers compatible with given requirements.

    Parameters
    ----------
    requirements : dict
        Requirements dictionary with keys like:
        - dtype: Required data type
        - needs_kvstore: Whether kvstore is required
        - needs_compression: Whether compression is required

    Returns
    -------
    list of str
        List of compatible driver names
    """
    compatible = []
    all_drivers = list_registered_drivers()

    for driver in all_drivers:
        capabilities = get_driver_capabilities(driver)

        # Check dtype compatibility
        if "dtype" in requirements:
            required_dtype = requirements["dtype"]
            if required_dtype not in capabilities["supported_dtypes"]:
                continue

        # Check kvstore requirement
        if requirements.get("needs_kvstore", False):
            if not capabilities["supports_kvstore"]:
                continue

        # Check compression requirement
        if requirements.get("needs_compression", False):
            if not capabilities["supports_compression"]:
                continue

        compatible.append(driver)

    return compatible


def _get_nested_list_shape(nested_list: list[Any]) -> list[int]:
    """Get shape of a nested list."""
    if not isinstance(nested_list, list) or not nested_list:
        return []

    shape = [len(nested_list)]

    # Check if first element is also a list
    if isinstance(nested_list[0], list):
        shape.extend(_get_nested_list_shape(nested_list[0]))

    return shape
