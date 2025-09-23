"""Builder classes for constructing TensorStore specifications."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from pydantic_tensorstore.core.context import Context
from pydantic_tensorstore.core.schema import Schema
from pydantic_tensorstore.core.spec import TensorStoreSpec
from pydantic_tensorstore.drivers.array import ArraySpec
from pydantic_tensorstore.drivers.zarr import ZarrSpec
from pydantic_tensorstore.drivers.n5 import N5Spec
from pydantic_tensorstore.kvstore.memory import MemoryKvStoreSpec
from pydantic_tensorstore.kvstore.file import FileKvStoreSpec
from pydantic_tensorstore.types.common import DataType, JsonObject


class SpecBuilder:
    """Builder for constructing TensorStore specifications.

    Provides a fluent interface for building complex specifications
    with method chaining and validation.

    Example:
        >>> builder = SpecBuilder()
        >>> spec = (builder
        ...     .driver("zarr")
        ...     .kvstore("memory")
        ...     .dtype("float32")
        ...     .shape([100, 200])
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._spec_data: dict[str, Any] = {}

    def driver(self, driver_name: str) -> SpecBuilder:
        """Set the driver."""
        self._spec_data["driver"] = driver_name
        return self

    def dtype(self, dtype: Union[str, DataType]) -> SpecBuilder:
        """Set the data type."""
        if "schema" not in self._spec_data:
            self._spec_data["schema"] = {}
        self._spec_data["schema"]["dtype"] = str(dtype)
        return self

    def shape(self, shape: list[int]) -> SpecBuilder:
        """Set the array shape."""
        if "schema" not in self._spec_data:
            self._spec_data["schema"] = {}
        if "domain" not in self._spec_data["schema"]:
            self._spec_data["schema"]["domain"] = {}
        self._spec_data["schema"]["domain"]["shape"] = shape
        return self

    def kvstore(
        self,
        kvstore: Union[str, dict[str, Any], MemoryKvStoreSpec, FileKvStoreSpec],
    ) -> SpecBuilder:
        """Set the key-value store."""
        if isinstance(kvstore, str):
            if kvstore == "memory":
                self._spec_data["kvstore"] = {"driver": "memory"}
            elif kvstore.startswith("/") or kvstore.startswith("file://"):
                path = kvstore.replace("file://", "")
                self._spec_data["kvstore"] = {"driver": "file", "path": path}
            else:
                # Assume it's a driver name
                self._spec_data["kvstore"] = {"driver": kvstore}
        elif isinstance(kvstore, dict):
            self._spec_data["kvstore"] = kvstore
        else:
            # Pydantic model
            self._spec_data["kvstore"] = kvstore.model_dump()
        return self

    def context(self, context: Union[dict[str, Any], Context]) -> SpecBuilder:
        """Set the context configuration."""
        if isinstance(context, dict):
            self._spec_data["context"] = context
        else:
            self._spec_data["context"] = context.model_dump()
        return self

    def chunk_shape(self, chunk_shape: list[int]) -> SpecBuilder:
        """Set the chunk shape."""
        if "schema" not in self._spec_data:
            self._spec_data["schema"] = {}
        if "chunk_layout" not in self._spec_data["schema"]:
            self._spec_data["schema"]["chunk_layout"] = {}
        self._spec_data["schema"]["chunk_layout"]["chunk_shape"] = chunk_shape
        return self

    def compression(self, compression: dict[str, Any]) -> SpecBuilder:
        """Set compression configuration (driver-dependent)."""
        driver = self._spec_data.get("driver")
        if driver == "zarr":
            if "metadata" not in self._spec_data:
                self._spec_data["metadata"] = {}
            self._spec_data["metadata"]["compressor"] = compression
        elif driver == "n5":
            if "metadata" not in self._spec_data:
                self._spec_data["metadata"] = {}
            self._spec_data["metadata"]["compression"] = compression
        else:
            # Generic schema codec
            if "schema" not in self._spec_data:
                self._spec_data["schema"] = {}
            self._spec_data["schema"]["codec"] = compression
        return self

    def path(self, path: str) -> SpecBuilder:
        """Set the path within the kvstore."""
        self._spec_data["path"] = path
        return self

    def metadata(self, metadata: dict[str, Any]) -> SpecBuilder:
        """Set driver-specific metadata."""
        self._spec_data["metadata"] = metadata
        return self

    def build(self) -> TensorStoreSpec:
        """Build the final specification."""
        return TensorStoreSpec.model_validate(self._spec_data)

    def build_dict(self) -> dict[str, Any]:
        """Build and return as dictionary."""
        return self._spec_data.copy()


class ArraySpecBuilder:
    """Builder specifically for Array driver specifications.

    Example:
        >>> builder = ArraySpecBuilder()
        >>> spec = (builder
        ...     .array([[1, 2], [3, 4]])
        ...     .dtype("int32")
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._spec_data: dict[str, Any] = {"driver": "array"}

    def array(self, array_data: Union[list[Any], np.ndarray]) -> ArraySpecBuilder:
        """Set the array data."""
        self._spec_data["array"] = array_data
        return self

    def dtype(self, dtype: Union[str, DataType]) -> ArraySpecBuilder:
        """Set the data type."""
        self._spec_data["dtype"] = str(dtype)
        return self

    def data_copy_concurrency(self, concurrency: Union[str, dict[str, Any]]) -> ArraySpecBuilder:
        """Set data copy concurrency."""
        self._spec_data["data_copy_concurrency"] = concurrency
        return self

    def context(self, context: Union[dict[str, Any], Context]) -> ArraySpecBuilder:
        """Set the context configuration."""
        if isinstance(context, dict):
            self._spec_data["context"] = context
        else:
            self._spec_data["context"] = context.model_dump()
        return self

    def build(self) -> ArraySpec:
        """Build the final specification."""
        return ArraySpec.model_validate(self._spec_data)


class ZarrSpecBuilder:
    """Builder specifically for Zarr driver specifications.

    Example:
        >>> builder = ZarrSpecBuilder()
        >>> spec = (builder
        ...     .kvstore("memory")
        ...     .path("my_array.zarr")
        ...     .chunks([64, 64])
        ...     .compression({"id": "blosc"})
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._spec_data: dict[str, Any] = {"driver": "zarr"}

    def kvstore(
        self,
        kvstore: Union[str, dict[str, Any], MemoryKvStoreSpec, FileKvStoreSpec],
    ) -> ZarrSpecBuilder:
        """Set the key-value store."""
        if isinstance(kvstore, str):
            if kvstore == "memory":
                self._spec_data["kvstore"] = {"driver": "memory"}
            else:
                self._spec_data["kvstore"] = {"driver": "file", "path": kvstore}
        elif isinstance(kvstore, dict):
            self._spec_data["kvstore"] = kvstore
        else:
            self._spec_data["kvstore"] = kvstore.model_dump()
        return self

    def path(self, path: str) -> ZarrSpecBuilder:
        """Set the path within the kvstore."""
        self._spec_data["path"] = path
        return self

    def chunks(self, chunks: list[int]) -> ZarrSpecBuilder:
        """Set the chunk shape."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["chunks"] = chunks
        return self

    def compression(self, compressor: dict[str, Any]) -> ZarrSpecBuilder:
        """Set the compression configuration."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["compressor"] = compressor
        return self

    def fill_value(self, fill_value: Union[int, float, str, bool]) -> ZarrSpecBuilder:
        """Set the fill value."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["fill_value"] = fill_value
        return self

    def order(self, order: str) -> ZarrSpecBuilder:
        """Set the memory order (C or F)."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["order"] = order
        return self

    def filters(self, filters: list[dict[str, Any]]) -> ZarrSpecBuilder:
        """Set the filter pipeline."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["filters"] = filters
        return self

    def build(self) -> ZarrSpec:
        """Build the final specification."""
        return ZarrSpec.model_validate(self._spec_data)


class N5SpecBuilder:
    """Builder specifically for N5 driver specifications.

    Example:
        >>> builder = N5SpecBuilder()
        >>> spec = (builder
        ...     .kvstore("/data/n5/")
        ...     .path("dataset")
        ...     .dimensions([1000, 1000, 100])
        ...     .block_size([64, 64, 64])
        ...     .data_type("uint16")
        ...     .compression({"type": "gzip"})
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._spec_data: dict[str, Any] = {"driver": "n5"}

    def kvstore(
        self,
        kvstore: Union[str, dict[str, Any], MemoryKvStoreSpec, FileKvStoreSpec],
    ) -> N5SpecBuilder:
        """Set the key-value store."""
        if isinstance(kvstore, str):
            if kvstore == "memory":
                self._spec_data["kvstore"] = {"driver": "memory"}
            else:
                self._spec_data["kvstore"] = {"driver": "file", "path": kvstore}
        elif isinstance(kvstore, dict):
            self._spec_data["kvstore"] = kvstore
        else:
            self._spec_data["kvstore"] = kvstore.model_dump()
        return self

    def path(self, path: str) -> N5SpecBuilder:
        """Set the path within the kvstore."""
        self._spec_data["path"] = path
        return self

    def dimensions(self, dimensions: list[int]) -> N5SpecBuilder:
        """Set the array dimensions."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["dimensions"] = dimensions
        return self

    def block_size(self, block_size: list[int]) -> N5SpecBuilder:
        """Set the block size."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["blockSize"] = block_size
        return self

    def data_type(self, data_type: str) -> N5SpecBuilder:
        """Set the N5 data type."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["dataType"] = data_type
        return self

    def compression(self, compression: dict[str, Any]) -> N5SpecBuilder:
        """Set the compression configuration."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["compression"] = compression
        return self

    def axes(self, axes: list[str]) -> N5SpecBuilder:
        """Set the axis labels."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["axes"] = axes
        return self

    def resolution(self, resolution: list[float]) -> N5SpecBuilder:
        """Set the resolution."""
        if "metadata" not in self._spec_data:
            self._spec_data["metadata"] = {}
        self._spec_data["metadata"]["resolution"] = resolution
        return self

    def build(self) -> N5Spec:
        """Build the final specification."""
        return N5Spec.model_validate(self._spec_data)