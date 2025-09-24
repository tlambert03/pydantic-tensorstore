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

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pydantic-tensorstore")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


from pydantic_tensorstore._types import DataType, OpenMode
from pydantic_tensorstore._validators import validate_spec
from pydantic_tensorstore.core import (
    ChunkedTensorStoreKvStoreAdapterSpec,
    Context,
    IndexTransform,
    Schema,
)
from pydantic_tensorstore.drivers import (
    ArraySpec,
    Codec,
    N5Spec,
    TensorStoreSpec,
    Zarr2Spec,
    Zarr3Spec,
)
from pydantic_tensorstore.kvstore import FileKvStore, KvStore, MemoryKvStore, S3KvStore

# FIXME: deal with circular references to Codecs
Schema.model_rebuild()

__all__ = [
    "ArraySpec",
    "ChunkedTensorStoreKvStoreAdapterSpec",
    "Codec",
    "Context",
    "DataType",
    "FileKvStore",
    "IndexTransform",
    "KvStore",
    "MemoryKvStore",
    "N5Spec",
    "OpenMode",
    "S3KvStore",
    "Schema",
    "TensorStoreSpec",
    "Zarr2Spec",
    "Zarr3Spec",
    "validate_spec",
]
