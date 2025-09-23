"""Pydantic models for TensorStore specifications.

This package provides type-safe, validated models for TensorStore specifications,
built with Pydantic v2 for excellent IDE support and runtime validation.

Example:
    >>> from pydantic_tensorstore import TensorStoreSpec
    >>> spec = TensorStoreSpec.model_validate(
    ...     {"driver": "array", "array": [[1, 2], [3, 4]], "dtype": "float32"}
    ... )
    >>> print(spec.driver)
    array
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "TensorStore Pydantic Team"
__email__ = "noreply@example.com"

from pydantic_tensorstore.core.context import Context
from pydantic_tensorstore.core.schema import Schema
from pydantic_tensorstore.core.spec import TensorStoreSpec
from pydantic_tensorstore.core.transform import IndexTransform

# Driver specs
from pydantic_tensorstore.drivers.array import ArraySpec
from pydantic_tensorstore.drivers.n5 import N5Spec
from pydantic_tensorstore.drivers.zarr import ZarrSpec
from pydantic_tensorstore.drivers.zarr3 import Zarr3Spec
from pydantic_tensorstore.kvstore.file import FileKvStoreSpec

# KvStore specs
from pydantic_tensorstore.kvstore.memory import MemoryKvStoreSpec
from pydantic_tensorstore.types.common import DataType, OpenMode

__all__ = [
    # Drivers
    "ArraySpec",
    "Context",
    # Types
    "DataType",
    "FileKvStoreSpec",
    "IndexTransform",
    # KvStore
    "MemoryKvStoreSpec",
    "N5Spec",
    "OpenMode",
    "Schema",
    # Core
    "TensorStoreSpec",
    "Zarr3Spec",
    "ZarrSpec",
]
