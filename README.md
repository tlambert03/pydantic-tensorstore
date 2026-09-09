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
- **No IDE support**: Without proper types, IDEs can't provide autocomplete,
  validation, or refactoring support

**pydantic-tensorstore** solves these issues by providing:

- [x] **Full type safety** with Pydantic v2 models
- [x] **Excellent IDE support** with autocomplete and validation
- [x] **Clear, actionable error messages** when specifications are invalid
- [x] **Runtime validation** with detailed error reporting
- [x] **Seamless conversion** to native TensorStore specs
- [x] **Documentation** embedded in the type system (field descriptions state
  TensorStore's defaults)

## Quick Example

Instead of wrestling with raw dictionaries:

```python
# Raw TensorStore (no types, no validation, no IDE support)
import tensorstore as ts

spec = {
    "driver": "zarr",
    "kvstore": {"driver": "file", "path": "/data/"},
    "metadata": {
        "chunks": [64, 64],
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5},
        "dtype": "<f4",  # Is this right? 🤔
    },
}
# Hope it works! 🤞
```

Use type-safe, validated specifications with IDE autocompletion:

```python
#  pydantic-tensorstore (full types, validation, IDE support)
import pydantic_tensorstore as pts

spec = pts.Zarr2Spec(
    kvstore=pts.MemoryKvStore(),
    metadata=pts.Zarr2Metadata(
        chunks=[64, 64],
        compressor=pts.Zarr2CompressorBlosc(cname="lz4", clevel=5),
        dtype="<f4",
    ),
)

# Convert to native TensorStore when needed
ts_spec = spec.to_tensorstore()  # requires tensorstore to be installed
```

To cast any dict (or JSON string, or `tensorstore.Spec`) to a validated spec:

```python
from pydantic_tensorstore import validate_spec

validated_spec = validate_spec(raw_dict)
```

## Installation

```bash
pip install pydantic-tensorstore

# with a compatible version of tensorstore installed for .to_tensorstore() support
pip install 'pydantic-tensorstore[tensorstore]'
```

## TensorStore compatibility

TensorStore's JSON spec is defined by the
[schema files in its repository](https://github.com/google/tensorstore/search?q=filename%3Aschema.yml).
The models here are checked field-by-field against those files for a single
pinned release, exposed as `pydantic_tensorstore.TENSORSTORE_VERSION`.

- **Verified against**: tensorstore `0.1.85` (`TENSORSTORE_VERSION`)
- **Tested with**: tensorstore `>= 0.1.68` (`MIN_TENSORSTORE_VERSION`). CI runs
  the full suite against the oldest, a midpoint, the pinned, and the newest
  release. Nearly everything the models can express already works on 0.1.68;
  the dozen fields and data types TensorStore added later carry a `since`
  marker.
- **Version check**: `spec.required_tensorstore_version()` reports the oldest
  release a spec needs, and `to_tensorstore()` raises
  `UnsupportedTensorStoreVersionError` naming the offending fields when the
  installed tensorstore is older (pass `check_version=False` to skip).
- **Drift**: a weekly CI job re-checks the models against the newest
  tensorstore release and opens an issue when something changes.

```python
>>> spec = pts.Zarr3Spec(kvstore="memory://", open_as_void=True)
>>> spec.required_tensorstore_version()
'0.1.85'
>>> spec.to_tensorstore()  # with tensorstore 0.1.68 installed
UnsupportedTensorStoreVersionError: installed tensorstore 0.1.68 does not support:
  open_as_void: requires tensorstore >= 0.1.85
```

Two design rules keep the models forward-compatible:

- **Optional fields default to `None`**, meaning "let TensorStore decide".
  TensorStore's own default is documented in the field description. Dumping a
  spec emits only the fields you set (an explicit `None` is kept as `null`,
  since TensorStore gives `null` meaning for e.g. `compressor`).
- **KvStore URLs pass through.** `file://`, `memory://`, `s3://`, `gs://` and
  `http(s)://` URLs are parsed into models; any other URL string (including
  pipelines such as `"memory://a.zip|zip:"`) is kept verbatim and handed to
  TensorStore unchanged.

## Known differences from TensorStore

A few deliberate or documented divergences, so they don't surprise you:

- **Partial specs are rejected.** TensorStore accepts `{"driver": "zarr3"}` with no
  `kvstore`, to be filled in later via `ts.Spec.update()` or an `open()` argument.
  The models require `kvstore` (and `cast`'s `dtype`), trading that pattern for a
  clear "you forgot the kvstore" error. Specs produced by a real store always
  include it.
- **Non-finite numbers become strings in JSON.** `model_dump_json()` writes
  `"NaN"`, `"Infinity"` and `"-Infinity"`, which TensorStore reads back as floats.
  Python-mode `model_dump()` keeps the float. Note that re-validating that JSON
  gives you the string back, since `fill_value` is untyped.
- **`schema` is spelled `schema_` on the model,** because `schema` collides with a
  Pydantic attribute. Both work at runtime, but mypy only accepts `schema_=`;
  `validate_spec()` takes plain `"schema"` in a dict either way.
- **Arrays are not silently truncated.** `ArraySpec` refuses a `dtype` that would
  lose data, rather than quietly rounding.

## Coverage

Every driver, kvstore, and context resource documented in tensorstore
`TENSORSTORE_VERSION` has a model.

| Kind | Models |
|---|---|
| Chunked drivers | `Zarr2Spec` (`zarr`/`zarr2`), `Zarr3Spec`, `N5Spec`, `NeuroglancerPrecomputedSpec` |
| Image drivers | `AvifSpec`, `BmpSpec`, `JpegSpec`, `PngSpec`, `TiffSpec`, `WebpSpec` |
| Other drivers | `ArraySpec`, `AutoSpec`, `CastSpec`, `DownsampleSpec`, `JsonSpec`, `StackSpec` |
| KvStores | `FileKvStore`, `MemoryKvStore`, `S3KvStore`, `GCSKvStore`, `HTTPKvStore`, `TsGrpcKvStore` |
| KvStore adapters | `OcdbtKvStore`, `ZipKvStore`, `KvStackKvStore`, `NeuroglancerUint64ShardedKvStore`, `Zarr3ShardingIndexedKvStore` |
| Codecs | `Zarr2Codec`, `Zarr3Codec` (+ every zarr3 codec), `N5Codec`, `NeuroglancerPrecomputedCodec` |
| Core | `Schema`, `ChunkLayout`, `IndexDomain`, `IndexTransform`, `Context` (+ every context resource), `Unit` |

`TensorStoreSpec` and `KvStore` are discriminated unions over all of the above;
`validate_spec()` and `validate_kvstore()` parse into them.

## Development

Tests compare the models against the vendored schema files in `tests/ts_schema/`
and, when tensorstore is installed, round-trip every spec through
`tensorstore.Spec`. To move to a newer tensorstore release:

```bash
uv run scripts/update_ts_schema.py --latest   # refresh tests/ts_schema/
# bump TENSORSTORE_VERSION in src/pydantic_tensorstore/__init__.py
uv run pytest tests/test_schema_conformance.py  # shows exactly what changed
```
