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


# Import all types, enums and utilities
# Import all core classes and types
from pydantic_tensorstore._core.chunk_layout import ChunkLayout, ChunkLayoutGrid
from pydantic_tensorstore._core.codec import CodecBase
from pydantic_tensorstore._core.context import (
    CachePool,
    Context,
    DataCopyConcurrency,
    FileIOConcurrency,
    HTTPConcurrency,
)
from pydantic_tensorstore._core.schema import Schema
from pydantic_tensorstore._core.spec import (
    BaseSpec,
    CacheRevalidationBound,
    ChunkedTensorStoreKvStoreAdapterSpec,
    TensorStoreKvStoreAdapterSpec,
)
from pydantic_tensorstore._core.transform import (
    ImplicitBound,
    IndexDomain,
    IndexInterval,
    IndexTransform,
    IntOrInf,
    OutputIndexMap,
)

# Import all driver specs and related classes
from pydantic_tensorstore._drivers import Codec, TensorStoreSpec

# Import Array-specific classes
from pydantic_tensorstore._drivers.array import ArraySpec

# Import N5-specific classes
from pydantic_tensorstore._drivers.n5 import (
    VALID_N5_DTYPES,
    N5Codec,
    N5Compression,
    N5DataType,
    N5Metadata,
    N5Spec,
)

# Import Neuroglancer-specific classes
from pydantic_tensorstore._drivers.neuroglancer_precomputed import (
    VALID_NEUROGLANCER_DTYPES,
    NeuroglancerDataType,
    NeuroglancerMultiscaleMetadata,
    NeuroglancerPrecomputedCodec,
    NeuroglancerPrecomputedSpec,
    NeuroglancerScaleMetadata,
    NeuroglancerShardingSpec,
)

# Import Zarr v2-specific classes
from pydantic_tensorstore._drivers.zarr import (
    Zarr2Codec,
    Zarr2Compressor,
    Zarr2CompressorBlosc,
    Zarr2CompressorBz2,
    Zarr2CompressorZlib,
    Zarr2CompressorZstd,
    Zarr2DataType,
    Zarr2Metadata,
    Zarr2SimpleDataType,
    Zarr2Spec,
    Zarr2StructuredDataType,
)

# Import Zarr v3-specific classes
from pydantic_tensorstore._drivers.zarr3 import (
    VALID_ZARR3_DTYPES,
    Zarr3ChunkConfiguration,
    Zarr3ChunkGrid,
    Zarr3ChunkKeyEncoding,
    Zarr3Codec,
    Zarr3CodecBlosc,
    Zarr3CodecBytes,
    Zarr3CodecChain,
    Zarr3CodecCRC32C,
    Zarr3CodecGzip,
    Zarr3CodecShardingIndexed,
    Zarr3CodecTranspose,
    Zarr3CodecZstd,
    Zarr3DataType,
    Zarr3Metadata,
    Zarr3SingleCodec,
    Zarr3Spec,
)

# Import KvStore classes
from pydantic_tensorstore._kvstore import (
    BaseKvStore,
    FileKvStore,
    KvStore,
    MemoryKvStore,
    S3KvStore,
)
from pydantic_tensorstore._types import (
    ChunkShape,
    ContextResource,
    ContextResourceName,
    DataType,
    DomainShape,
    DriverName,
    OpenMode,
    ReadWriteMode,
    Shape,
    Unit,
)
from pydantic_tensorstore._validators import validate_spec

# FIXME: deal with circular references to Codecs
Schema.model_rebuild()

__all__ = [
    "VALID_N5_DTYPES",
    "VALID_NEUROGLANCER_DTYPES",
    "VALID_ZARR3_DTYPES",
    "ArraySpec",
    "BaseKvStore",
    "BaseSpec",
    "CachePool",
    "CacheRevalidationBound",
    "ChunkLayout",
    "ChunkLayoutGrid",
    "ChunkShape",
    "ChunkedTensorStoreKvStoreAdapterSpec",
    "Codec",
    "CodecBase",
    "Context",
    "ContextResource",
    "ContextResourceName",
    "DataCopyConcurrency",
    "DataType",
    "DomainShape",
    "DriverName",
    "FileIOConcurrency",
    "FileKvStore",
    "HTTPConcurrency",
    "ImplicitBound",
    "IndexDomain",
    "IndexInterval",
    "IndexTransform",
    "IntOrInf",
    "KvStore",
    "MemoryKvStore",
    "N5Codec",
    "N5Compression",
    "N5DataType",
    "N5Metadata",
    "N5Spec",
    "NeuroglancerDataType",
    "NeuroglancerMultiscaleMetadata",
    "NeuroglancerPrecomputedCodec",
    "NeuroglancerPrecomputedSpec",
    "NeuroglancerScaleMetadata",
    "NeuroglancerShardingSpec",
    "OpenMode",
    "OutputIndexMap",
    "ReadWriteMode",
    "S3KvStore",
    "Schema",
    "Shape",
    "TensorStoreKvStoreAdapterSpec",
    "TensorStoreSpec",
    "Unit",
    "Zarr2Codec",
    "Zarr2Compressor",
    "Zarr2CompressorBlosc",
    "Zarr2CompressorBz2",
    "Zarr2CompressorZlib",
    "Zarr2CompressorZstd",
    "Zarr2DataType",
    "Zarr2Metadata",
    "Zarr2SimpleDataType",
    "Zarr2Spec",
    "Zarr2StructuredDataType",
    "Zarr3ChunkConfiguration",
    "Zarr3ChunkGrid",
    "Zarr3ChunkKeyEncoding",
    "Zarr3Codec",
    "Zarr3CodecBlosc",
    "Zarr3CodecBytes",
    "Zarr3CodecCRC32C",
    "Zarr3CodecChain",
    "Zarr3CodecGzip",
    "Zarr3CodecShardingIndexed",
    "Zarr3CodecTranspose",
    "Zarr3CodecZstd",
    "Zarr3DataType",
    "Zarr3Metadata",
    "Zarr3SingleCodec",
    "Zarr3Spec",
    "validate_spec",
]
