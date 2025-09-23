"""Pydantic models for TensorStore specifications.

This package provides type-safe, validated models for TensorStore specifications,
built with Pydantic v2 for excellent IDE support and runtime validation.

Example:
    >>> from pydantic_tensorstore import validate_spec
    >>> spec = validate_spec(
    ...     {"driver": "array", "array": [[1, 2], [3, 4]], "dtype": "float32"}
    ... )
    >>> print(spec.driver)
    array
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pydantic-tensorstore")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


from pydantic_tensorstore._types import DataType, OpenMode
from pydantic_tensorstore.core import (
    BaseDriverSpec,
    Context,
    IndexTransform,
    Schema,
    TensorStoreSpec,
)
from pydantic_tensorstore.drivers import ArraySpec, N5Spec, Zarr3Spec, ZarrSpec
from pydantic_tensorstore.kvstore import FileKvStoreSpec, MemoryKvStoreSpec
from pydantic_tensorstore.validators import validate_spec

__all__ = [
    "ArraySpec",
    "BaseDriverSpec",
    "Context",
    "DataType",
    "FileKvStoreSpec",
    "IndexTransform",
    "MemoryKvStoreSpec",
    "N5Spec",
    "OpenMode",
    "Schema",
    "TensorStoreSpec",
    "Zarr3Spec",
    "ZarrSpec",
    "validate_spec",
]
