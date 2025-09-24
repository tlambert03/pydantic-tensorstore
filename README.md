# pydantic-tensorstore

[![License](https://img.shields.io/pypi/l/pydantic-tensorstore.svg?color=green)](https://github.com/tlambert03/pydantic-tensorstore/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pydantic-tensorstore.svg?color=green)](https://pypi.org/project/pydantic-tensorstore)
[![Python Version](https://img.shields.io/pypi/pyversions/pydantic-tensorstore.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/pydantic-tensorstore/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/pydantic-tensorstore/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/pydantic-tensorstore/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/pydantic-tensorstore)

*Type-safe, validated TensorStore specifications with Pydantic*

## Motivation

[TensorStore](https://github.com/google/tensorstore) is an exceptional C++ and
Python library for reading and writing large multi-dimensional arrays. It
supports numerous storage formats (Zarr, N5, Neuroglancer Precomputed) and
backends (local files, cloud storage, memory), making it incredibly powerful for
scientific computing and data analysis.

However, TensorStore has some pain points that this library attempts to address:

- **Poor type hinting**: TensorStore specifications are typically created as
  dictionaries with minimal type information, making it difficult to know what
  fields are available or required
- **Complex documentation**: Creating proper TensorStore JSON spec objects often
  requires constantly referencing web documentation to understand the various
  options and their relationships
- **Cryptic error messages**: When specifications are malformed, TensorStore
  errors can be difficult to interpret and debug
- **No IDE support**: Without proper types, IDEs can't provide autocomplete,
  validation, or refactoring support

**pydantic-tensorstore** solves these issues by providing:

- [x] **Full type safety** with Pydantic v2 models
- [x] **Excellent IDE support** with autocomplete and validation
- [x] **Clear, actionable error messages** when specifications are invalid
- [x] **Runtime validation** with detailed error reporting
- [x] **Seamless conversion** to native TensorStore specs
- [x] **Comprehensive documentation** embedded in the type system

## Quick Example

Instead of wrestling with raw dictionaries:

```python
# Raw TensorStore (no types, no validation, no IDE support)
import tensorstore as ts

spec = {
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
        "path": "/data/"
    },
    "metadata": {
        "chunks": [64, 64],
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5},
        "dtype": "<f4"  # Is this right? 🤔
    }
}
# Hope it works! 🤞
```

Use type-safe, validated specifications:

```python
#  pydantic-tensorstore (full types, validation, IDE support)
import pydantic_tensorstore as pts
from pydantic_tensorstore.drivers import zarr

spec = pts.Zarr2Spec(
    kvstore=pts.MemoryKvStore(),
    metadata=zarr.ZarrMetadata(
        chunks=[64, 64],
        compressor=zarr.Zarr2CompressorBlosc(cname="lz4", clevel=5),
        dtype="<f4",
    ),
)

# Convert to native TensorStore when needed
ts_spec = spec.to_tensorstore()  # requires tensorstore to be installed
```

To cast any dict to a validated spec:

```python
from pydantic_tensorstore import validate_spec

validated_spec = validate_spec(raw_dict)
```

## Installation

install from github for now

```bash
pip install git+https://github.com/tlambert03/pydantic-tensorstore
```

## Features

### Supported Drivers

- **Array**: In-memory arrays with NumPy integration
- **Zarr v2**: Full support for Zarr v2 format with all compression options
- **Zarr v3**: Support for the new Zarr v3 specification
- **N5**: N5 format support with compression and chunking
- **Neuroglancer Precomputed**: For neuroimaging workflows

### Supported Storage Backends

- **File**: Local filesystem storage
- **Memory**: In-memory storage for testing and caching
- **S3**: AWS S3 and S3-compatible storage

### Key Features

- **Discriminated unions**: Automatically parse the correct spec type based on
  the `driver` field
- **Validation**: Runtime validation with helpful error messages
- **Serialization**: JSON serialization and deserialization
- **IDE integration**: Full type hints and autocomplete support
- **Documentation**: Embedded field descriptions and examples
