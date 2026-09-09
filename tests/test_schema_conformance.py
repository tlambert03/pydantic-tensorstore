"""Diff our models against tensorstore's own JSON-schema YAML files.

The YAML files under `tests/ts_schema/` are vendored from google/tensorstore at the
release named in `tests/ts_schema/VERSION` (refresh with `scripts/update_ts_schema.py`).
Set `PTS_SCHEMA_DIR` to point at a different snapshot (used by the drift CI job).
"""

from __future__ import annotations

import enum
import os
import types
import typing
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin

import annotated_types
import pytest
import yaml
from conftest import skip_if_older_tensorstore

import pydantic_tensorstore as pts
from pydantic_tensorstore._core.context import _AwsCredentials

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import BaseModel
    from pydantic.fields import FieldInfo

SCHEMA_DIR = Path(os.environ.get("PTS_SCHEMA_DIR", Path(__file__).parent / "ts_schema"))

# ---------------------------------------------------------------------------
# schema loading
# ---------------------------------------------------------------------------


def _walk(node: Any, file: str, out: dict[str, tuple[dict, str]]) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("$id"), str):
            id_ = node["$id"]
            key = f"{file}{id_}" if id_.startswith("#") else id_
            out[key] = (node, file)
        for v in node.values():
            _walk(v, file, out)
    elif isinstance(node, list):
        for v in node:
            _walk(v, file, out)


def _load() -> tuple[dict[str, tuple[dict, str]], dict[str, dict]]:
    ids: dict[str, tuple[dict, str]] = {}
    files: dict[str, dict] = {}
    for path in sorted(SCHEMA_DIR.glob("*.yml")):
        data = yaml.safe_load(path.read_text())
        files[path.name] = data
        _walk(data, path.name, ids)
    return ids, files


IDS, FILES = _load()


def _ref_key(ref: str, file: str) -> str:
    return f"{file}{ref}" if ref.startswith("#") else ref


def _resolve(node: dict, file: str) -> tuple[dict[str, dict], set[str]]:
    """Flatten a schema node (following allOf/$ref) into (properties, required)."""
    props: dict[str, dict] = {}
    required: set[str] = set()
    if "$ref" in node and node["$ref"] in {_ref_key(node["$ref"], file), node["$ref"]}:
        key = _ref_key(node["$ref"], file)
        if key in IDS:
            p, r = _resolve(*IDS[key])
            props.update(p)
            required |= r
    for sub in node.get("allOf", []):
        p, r = _resolve(sub, file)
        props.update(p)
        required |= r
    for name, prop in (node.get("properties") or {}).items():
        merged = dict(props.get(name, {}))
        merged.update(prop)
        props[name] = merged
    required |= set(node.get("required") or [])
    return props, required


def _node_for(spec_id: str) -> tuple[dict, str]:
    """Look up `$id`, `$id.prop.subprop`, or `$id.prop[]` (array items)."""
    if spec_id in IDS:
        return IDS[spec_id]
    base = max((c for c in IDS if spec_id.startswith(c + ".")), key=len)
    node, file = IDS[base]
    for step in spec_id[len(base) + 1 :].split("."):
        props, _ = _resolve(node, file)
        node = props[step.removesuffix("[]")]
        if step.endswith("[]"):
            node = node["items"]
    return node, file


# ---------------------------------------------------------------------------
# registry: schema $id -> our model
# ---------------------------------------------------------------------------

MODEL_FOR_ID: dict[str, type[BaseModel]] = {
    # core
    "TensorStore": pts.BaseSpec,
    "TensorStoreKvStoreAdapter": pts.TensorStoreKvStoreAdapterSpec,
    "TensorStoreAdapter": pts.TensorStoreAdapterSpec,
    "ChunkedTensorStoreKvStoreAdapter": pts.ChunkedTensorStoreKvStoreAdapterSpec,
    "Schema": pts.Schema,
    "ChunkLayout": pts.ChunkLayout,
    "ChunkLayout/Grid": pts.ChunkLayoutGrid,
    "IndexDomain": pts.IndexDomain,
    "IndexTransform": pts.IndexTransform,
    "OutputIndexMap": pts.OutputIndexMap,
    "Context": pts.Context,
    "Context.cache_pool": pts.CachePool,
    "Context.data_copy_concurrency": pts.DataCopyConcurrency,
    "Context.file_io_concurrency": pts.FileIOConcurrency,
    "Context.file_io_mode": pts.FileIOMode,
    "Context.file_io_locking": pts.FileIOLocking,
    "Context.http_request_concurrency": pts.HTTPRequestConcurrency,
    "Context.http_request_retries": pts.RequestRetries,
    "Context.gcs_request_concurrency": pts.HTTPRequestConcurrency,
    "Context.gcs_request_retries": pts.RequestRetries,
    "Context.gcs_user_project": pts.GCSUserProject,
    "Context.experimental_gcs_rate_limiter": pts.RateLimiter,
    "Context.s3_request_concurrency": pts.HTTPRequestConcurrency,
    "Context.s3_request_retries": pts.RequestRetries,
    "Context.experimental_s3_rate_limiter": pts.RateLimiter,
    "Context.aws_credentials": _AwsCredentials,
    "Context.aws_credentials/anonymous": pts.AwsCredentialsAnonymous,
    "Context.aws_credentials/environment": pts.AwsCredentialsEnvironment,
    "Context.aws_credentials/imds": pts.AwsCredentialsImds,
    "Context.aws_credentials/default": pts.AwsCredentialsDefault,
    "Context.aws_credentials/profile": pts.AwsCredentialsProfile,
    "Context.aws_credentials/ecs": pts.AwsCredentialsEcs,
    "Context.memory_key_value_store": pts.MemoryKeyValueStore,
    "Context.ocdbt_coordinator": pts.OcdbtCoordinator,
    # drivers
    "driver/array": pts.ArraySpec,
    "driver/auto": pts.AutoSpec,
    "driver/cast": pts.CastSpec,
    "driver/downsample": pts.DownsampleSpec,
    "driver/stack": pts.StackSpec,
    "driver/json": pts.JsonSpec,
    "driver/avif": pts.AvifSpec,
    "driver/bmp": pts.BmpSpec,
    "driver/jpeg": pts.JpegSpec,
    "driver/png": pts.PngSpec,
    "driver/tiff": pts.TiffSpec,
    "driver/webp": pts.WebpSpec,
    "driver/n5": pts.N5Spec,
    "driver/n5.metadata": pts.N5Metadata,
    "driver/n5/Codec": pts.N5Codec,
    "driver/n5/Compression/raw": pts.N5CompressionRaw,
    "driver/n5/Compression/gzip": pts.N5CompressionGzip,
    "driver/n5/Compression/bzip2": pts.N5CompressionBzip2,
    "driver/n5/Compression/xz": pts.N5CompressionXZ,
    "driver/n5/Compression/zstd": pts.N5CompressionZstd,
    "driver/n5/Compression/blosc": pts.N5CompressionBlosc,
    "driver/neuroglancer_precomputed": pts.NeuroglancerPrecomputedSpec,
    "driver/neuroglancer_precomputed.multiscale_metadata": (
        pts.NeuroglancerMultiscaleMetadata
    ),
    "driver/neuroglancer_precomputed.scale_metadata": pts.NeuroglancerScaleMetadata,
    "driver/neuroglancer_precomputed/Codec": pts.NeuroglancerPrecomputedCodec,
    "driver/zarr2": pts.Zarr2Spec,
    "driver/zarr2.metadata": pts.Zarr2Metadata,
    "driver/zarr2/Codec": pts.Zarr2Codec,
    "driver/zarr2/Compressor/zlib": pts.Zarr2CompressorZlib,
    "driver/zarr2/Compressor/blosc": pts.Zarr2CompressorBlosc,
    "driver/zarr2/Compressor/bz2": pts.Zarr2CompressorBz2,
    "driver/zarr2/Compressor/zstd": pts.Zarr2CompressorZstd,
    "driver/zarr3": pts.Zarr3Spec,
    "driver/zarr3/Metadata": pts.Zarr3Metadata,
    "driver/zarr3/Metadata.chunk_grid": pts.Zarr3ChunkGrid,
    "driver/zarr3/Metadata.chunk_grid.configuration": pts.Zarr3ChunkConfiguration,
    "driver/zarr3/Metadata.attributes": pts.Zarr3Attributes,
    "driver/zarr3/ChunkKeyEncoding.default": pts.Zarr3ChunkKeyEncodingDefault,
    "driver/zarr3/ChunkKeyEncoding.default.configuration": (
        pts.Zarr3ChunkKeyEncodingConfig
    ),
    "driver/zarr3/ChunkKeyEncoding.v2": pts.Zarr3ChunkKeyEncodingV2,
    "driver/zarr3/ChunkKeyEncoding.v2.configuration": pts.Zarr3ChunkKeyEncodingConfig,
    "driver/zarr3/Codec": pts.Zarr3Codec,
    "driver/zarr3/Codec/bytes": pts.Zarr3CodecBytes,
    "driver/zarr3/Codec/bytes.configuration": pts.Zarr3BytesConfig,
    "driver/zarr3/Codec/sharding_indexed": pts.Zarr3CodecShardingIndexed,
    "driver/zarr3/Codec/sharding_indexed.configuration": (
        pts.Zarr3ShardingIndexedConfig
    ),
    "driver/zarr3/Codec/transpose": pts.Zarr3CodecTranspose,
    "driver/zarr3/Codec/transpose.configuration": pts.Zarr3TransposeConfig,
    "driver/zarr3/Codec/crc32c": pts.Zarr3CodecCRC32C,
    "driver/zarr3/Codec/crc32c.configuration": pts.Zarr3CRC32CConfig,
    "driver/zarr3/Codec/gzip": pts.Zarr3CodecGzip,
    "driver/zarr3/Codec/gzip.configuration": pts.Zarr3GzipConfig,
    "driver/zarr3/Codec/blosc": pts.Zarr3CodecBlosc,
    "driver/zarr3/Codec/blosc.configuration": pts.Zarr3BloscConfig,
    "driver/zarr3/Codec/zstd": pts.Zarr3CodecZstd,
    "driver/zarr3/Codec/zstd.configuration": pts.Zarr3ZstdConfig,
    # kvstores
    "KvStore": pts.BaseKvStore,
    "KvStoreAdapter": pts.KvStoreAdapter,
    "kvstore/file": pts.FileKvStore,
    "kvstore/memory": pts.MemoryKvStore,
    "kvstore/s3": pts.S3KvStore,
    "kvstore/gcs": pts.GCSKvStore,
    "kvstore/http": pts.HTTPKvStore,
    "kvstore/kvstack": pts.KvStackKvStore,
    "kvstore/kvstack.layers[]": pts.KvStackLayer,
    "kvstore/ocdbt": pts.OcdbtKvStore,
    "kvstore/ocdbt.config": pts.OcdbtConfig,
    "kvstore/ocdbt/Compression/zstd": pts.OcdbtZstdCompression,
    "kvstore/zip": pts.ZipKvStore,
    "kvstore/tsgrpc_kvstore": pts.TsGrpcKvStore,
    "kvstore/neuroglancer_uint64_sharded": pts.NeuroglancerUint64ShardedKvStore,
    "kvstore/neuroglancer_uint64_sharded/ShardingSpec": pts.NeuroglancerShardingSpec,
    "kvstore/zarr3_sharding_indexed": pts.Zarr3ShardingIndexedKvStore,
}

# $ids that are deliberately not modelled as objects (scalars, unions, abstract bases)
NON_OBJECT_IDS = {
    "dtype",  # tested in test_dtypes.py
    "Unit",
    "IndexInterval",
    "CacheRevalidationBound",
    "ContextResource",
    "Codec",
    "DownsampleMethod",
    "driver/zarr3/DataType",
    "driver/zarr3/CodecChain",
    "driver/zarr3/SingleCodec",
    "driver/zarr3/ChunkKeyEncoding",
    "driver/zarr2/Compressor",
    "driver/n5/Compression",
    "Context.file_io_sync",
}

# schema properties we knowingly do not model, per $id
KNOWN_GAPS: dict[str, set[str]] = {
    # abstract bases: the discriminator is declared (as a Literal) on each subclass
    "TensorStore": {"driver"},
    "TensorStoreKvStoreAdapter": {"driver"},
    "TensorStoreAdapter": {"driver"},
    "ChunkedTensorStoreKvStoreAdapter": {"driver"},
    "KvStore": {"driver"},
    "KvStoreAdapter": {"driver"},
    "Context.aws_credentials": {"type"},
}

# model fields with no counterpart in the schema, per $id
KNOWN_EXTRAS: dict[str, set[str]] = {
    # every typed `Context` field must correspond to a `Context.<name>` schema id
    "Context": {id_[len("Context.") :] for id_ in IDS if id_.startswith("Context.")},
    # documented as an opaque object, but tensorstore accepts (and validates) `level`
    "driver/zarr3/Codec/gzip.configuration": {"level"},
}

# examples in the tensorstore repo that tensorstore itself rejects
INVALID_EXAMPLES = {
    "driver/stack[0]",  # uses scalar `input_inclusive_min` / object `output`
}

# fields the schema requires that we leave optional
KNOWN_LOOSER: dict[str, set[str]] = {
    # tensorstore itself omits `dtype` when an array is the `base` of an adapter
    "driver/array": {"dtype"},
}

# fields we require although the schema does not (all have no usable default)
KNOWN_STRICTER: dict[str, set[str]] = {
    "kvstore/s3": {"bucket"},  # schema default "" is not a usable bucket
    "driver/zarr3": {"kvstore"},  # required via KvStoreAdapter, inherited
}

# ---------------------------------------------------------------------------
# helpers on our side
# ---------------------------------------------------------------------------


def _fields(model: type[BaseModel]) -> dict[str, FieldInfo]:
    return {f.alias or name: f for name, f in model.model_fields.items()}


def _iter_annotations(tp: Any) -> Iterator[Any]:
    """Yield every leaf annotation, descending into Unions and Annotated."""
    origin = get_origin(tp)
    if origin is Annotated:
        yield tp
        yield from _iter_annotations(get_args(tp)[0])
    elif origin in (typing.Union, types.UnionType):
        for arg in get_args(tp):
            yield from _iter_annotations(arg)
    else:
        yield tp


def _literal_values(field: FieldInfo) -> set[Any] | None:
    values: set[Any] = set()
    found = False
    for ann in _iter_annotations(field.annotation):
        if get_origin(ann) is Literal:
            found = True
            for v in get_args(ann):
                values.add(v.value if isinstance(v, enum.Enum) else v)
    return values if found else None


def _bounds(field: FieldInfo) -> tuple[Any, Any]:
    lo = hi = None
    metas: list[Any] = list(field.metadata)
    for ann in _iter_annotations(field.annotation):
        if get_origin(ann) is Annotated:
            metas += list(get_args(ann)[1:])
    for m in metas:
        if isinstance(m, annotated_types.Interval):
            lo = m.ge if m.ge is not None else lo
            hi = m.le if m.le is not None else hi
        elif isinstance(m, annotated_types.Ge):
            lo = m.ge
        elif isinstance(m, annotated_types.Le):
            hi = m.le
        elif isinstance(m, annotated_types.Gt):
            lo = m.gt + 1
        elif isinstance(m, annotated_types.Lt):
            hi = m.lt - 1
    return lo, hi


def _schema_consts(prop: dict) -> set[Any] | None:
    if "const" in prop:
        return {prop["const"]}
    if "enum" in prop:
        return set(prop["enum"])
    if "oneOf" in prop and all("const" in o for o in prop["oneOf"]):
        return {o["const"] for o in prop["oneOf"]}
    return None


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_vendored_version_matches_package() -> None:
    assert (SCHEMA_DIR / "VERSION").read_text().strip() == pts.TENSORSTORE_VERSION


def test_every_driver_and_kvstore_is_modelled() -> None:
    object_ids = {
        id_
        for id_, (node, _) in IDS.items()
        if not id_.startswith(("TensorStoreUrl", "KvStoreUrl"))
        and not id_.startswith("tensorstore_")  # file-local '#...' definitions
        and not id_.startswith("docs_")
        and (node.get("type") == "object" or "allOf" in node or "properties" in node)
    }
    unmodelled = object_ids - set(MODEL_FOR_ID) - NON_OBJECT_IDS
    assert not unmodelled, f"schema ids with no model: {sorted(unmodelled)}"


@pytest.mark.parametrize("spec_id", sorted(MODEL_FOR_ID), ids=str)
def test_model_matches_schema(spec_id: str) -> None:
    model = MODEL_FOR_ID[spec_id]
    node, file = _node_for(spec_id)
    props, required = _resolve(node, file)
    props = {k: v for k, v in props.items() if not k.startswith("<")}
    ours = _fields(model)

    missing = set(props) - set(ours) - KNOWN_GAPS.get(spec_id, set())
    assert not missing, f"{spec_id}: schema properties not modelled: {sorted(missing)}"

    extra = set(ours) - set(props) - KNOWN_EXTRAS.get(spec_id, set())
    assert not extra, f"{spec_id}: model fields not in schema: {sorted(extra)}"

    problems: list[str] = []
    for name in sorted(set(props) & set(ours)):
        prop, field = props[name], ours[name]
        consts = _schema_consts(prop)
        literals = _literal_values(field)
        if consts is not None and literals is not None and consts != literals:
            problems.append(f"{name}: schema values {consts} != ours {literals}")
        if "default" in prop and prop["default"] is not None and literals is None:
            if field.default is not None and field.is_required() is False:
                problems.append(
                    f"{name}: has default {field.default!r}; schema default "
                    f"{prop['default']!r} should be left to tensorstore (use None)"
                )
        lo, hi = _bounds(field)
        if "minimum" in prop and lo != prop["minimum"]:
            problems.append(f"{name}: minimum {prop['minimum']} != ours {lo}")
        if "maximum" in prop and hi != prop["maximum"]:
            problems.append(f"{name}: maximum {prop['maximum']} != ours {hi}")

    ours_required = {n for n, f in ours.items() if f.is_required()}
    ours_satisfied = {
        n for n, f in ours.items() if f.is_required() or f.default is not None
    }
    unmet = required - ours_satisfied - KNOWN_GAPS.get(spec_id, set())
    if unmet := unmet - KNOWN_LOOSER.get(spec_id, set()):
        problems.append(f"schema requires {sorted(unmet)} but they are optional here")
    stricter = ours_required - required - KNOWN_STRICTER.get(spec_id, set())
    if stricter:
        problems.append(f"required here but optional in schema: {sorted(stricter)}")

    assert not problems, f"{spec_id}:\n  " + "\n  ".join(problems)


def _examples() -> Iterator[Any]:
    for id_, (node, _file) in IDS.items():
        if id_ not in MODEL_FOR_ID:
            continue
        for i, ex in enumerate(node.get("examples") or []):
            if isinstance(ex, dict) and f"{id_}[{i}]" not in INVALID_EXAMPLES:
                yield pytest.param(id_, ex, id=f"{id_}[{i}]")


@pytest.mark.parametrize(("spec_id", "example"), list(_examples()))
def test_schema_examples_validate(spec_id: str, example: Any) -> None:
    """Every example in tensorstore's docs validates, and round-trips identically."""
    model = MODEL_FOR_ID[spec_id]
    ours = model.model_validate(example)
    ts = pytest.importorskip("tensorstore")
    if issubclass(model, pts.BaseSpec):
        parse = ts.Spec
    elif issubclass(model, pts.BaseKvStore):
        parse = ts.KvStore.Spec
    else:
        return
    try:
        expected = parse(example)
    except ValueError as e:
        skip_if_older_tensorstore(e)
    assert parse(ours.model_dump(mode="json")) == expected
