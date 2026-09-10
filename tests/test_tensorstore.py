import json
import math
from pathlib import Path

import pytest
from conftest import PINNED_VERSION, TS_VERSION, skip_if_older_tensorstore
from pydantic import TypeAdapter

import pydantic_tensorstore as pts
from pydantic_tensorstore import validate_spec

try:
    import tensorstore as ts
except ImportError:
    pytest.skip("tensorstore not installed", allow_module_level=True)


# Test cases for round-trip validation
ROUND_TRIP_TEST_CASES = [
    # Array driver examples
    {
        "id": "array_basic",
        "spec": {
            "driver": "array",
            "array": [[1, 2, 3], [4, 5, 6]],
            "dtype": "int32",
        },
    },
    {
        "id": "array_with_transform",
        "spec": {
            "driver": "array",
            "array": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            "dtype": "float32",
            "transform": {
                "input_inclusive_min": [0, 0, 0],
                "input_exclusive_max": [2, 2, 2],
            },
        },
    },
    # Zarr v2 examples
    {
        "id": "zarr_memory_basic",
        "spec": {
            "driver": "zarr",
            "dtype": "uint16",
            "kvstore": {"driver": "memory"},
            "create": True,
            "metadata": {"shape": [256, 256]},
        },
    },
    {
        "id": "zarr_memory_with_metadata",
        "spec": {
            "driver": "zarr",
            "kvstore": {"driver": "memory"},
            "metadata": {
                "chunks": [64, 64],
                "shape": [1000, 1000],
                "dtype": "<f4",
                "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5},
                "order": "C",
                "fill_value": 0.0,
            },
        },
    },
    {
        "id": "zarr_memory_structured_dtype",
        "spec": {
            "driver": "zarr",
            "kvstore": {"driver": "memory"},
            "metadata": {
                "chunks": [100],
                "shape": [1000],
                "dtype": [["r", "|u1"], ["g", "|u1"], ["b", "|u1"]],
                "compressor": {"id": "zlib", "level": 6},
            },
            "field": "g",
        },
    },
    {
        "id": "zarr_file_with_path",
        "spec": {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": "test_zarr"},
            "path": "dataset.zarr",
            "schema": {"domain": {"shape": [64, 64, 64]}},
            "metadata": {
                "chunks": [8, 8, 8],
                "dtype": ">i2",
                "compressor": {"id": "zstd", "level": 3},
                "dimension_separator": "/",
            },
        },
    },
    # Zarr v3 examples
    {
        "id": "zarr3_basic",
        "spec": {
            "driver": "zarr3",
            "kvstore": {"driver": "memory"},
            "metadata": {
                "shape": [100, 200],
                "data_type": "uint16",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [50, 100]},
                },
            },
        },
    },
    {
        "id": "zarr3_file",
        "spec": {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": "zarr_test"},
            "metadata": {
                "shape": [3, 4, 5],
                "data_type": "float32",
                "chunk_key_encoding": {"name": "v2"},
            },
            "create": True,
            "delete_existing": True,
        },
    },
    {
        "id": "zarr3_with_codecs",
        "spec": {
            "driver": "zarr3",
            "kvstore": {"driver": "memory"},
            "metadata": {
                "shape": [1000, 500, 100],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [100, 100, 50]},
                },
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 3}},
                ],
                "fill_value": -1.0,
            },
        },
    },
    {
        "id": "zarr3_with_sharding",
        "spec": {
            "driver": "zarr3",
            "kvstore": {"driver": "memory"},
            "metadata": {
                "shape": [10000, 10000],
                "data_type": "uint8",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1000, 1000]},
                },
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": [100, 100],
                            "codecs": [
                                {"name": "bytes"},
                                {"name": "gzip", "configuration": {"level": 6}},
                            ],
                        },
                    },
                ],
            },
        },
    },
    # N5 examples
    {
        "id": "n5_basic",
        "spec": {
            "driver": "n5",
            "kvstore": {"driver": "memory"},
            "metadata": {
                "dimensions": [1000, 1000, 100],
                "blockSize": [64, 64, 32],
                "dataType": "uint16",
            },
        },
    },
    {
        "id": "n5_with_compression",
        "spec": {
            "driver": "n5",
            "kvstore": {"driver": "file", "path": "n5_test"},
            "path": "dataset",
            "metadata": {
                "dimensions": [2000, 2000, 200],
                "blockSize": [128, 128, 64],
                "dataType": "float32",
                "compression": {"type": "gzip", "level": 6},
            },
        },
    },
    {
        "id": "n5_bzip2_compression",
        "spec": {
            "driver": "n5",
            "kvstore": {"driver": "memory"},
            "metadata": {
                "dimensions": [500, 500],
                "blockSize": [64, 64],
                "dataType": "int32",
                "compression": {"type": "bzip2"},
            },
        },
    },
    # Neuroglancer examples
    {
        "id": "neuroglancer_basic",
        "spec": {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "memory"},
            "multiscale_metadata": {
                "type": "image",
                "data_type": "uint8",
                "num_channels": 1,
            },
            "scale_metadata": {
                "key": "1_1_1",
                "size": (1024, 1024, 100),
                "chunk_size": (64, 64, 16),
                "resolution": (1.0, 1.0, 1.0),
                "encoding": "raw",
            },
        },
    },
    {
        "id": "neuroglancer_jpeg",
        "spec": {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "memory"},
            "dtype": "uint8",
            "schema": {"domain": {"shape": [512, 512, 200, 3]}},
            "scale_metadata": {
                "key": "2_2_2",
                "size": (512, 512, 200),
                "chunk_size": (128, 128, 32),
                "resolution": (2.0, 2.0, 2.0),
                "encoding": "jpeg",
                "jpeg_quality": 85,
            },
        },
    },
    {
        "id": "neuroglancer_segmentation",
        "spec": {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "memory"},
            "multiscale_metadata": {
                "type": "segmentation",
                "data_type": "uint32",
                "num_channels": 1,
            },
            "scale_metadata": {
                "key": "1_1_1",
                "size": (2048, 2048, 512),
                "chunk_size": (128, 128, 64),
                "resolution": (8.0, 8.0, 8.0),
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": (8, 8, 8),
            },
        },
    },
    # TIFF examples (validation-only, no actual creation)
    {
        "id": "tiff_basic",
        "spec": {
            "driver": "tiff",
            "kvstore": {"driver": "memory"},
            "path": "image.tiff",
        },
        "skip_creation": True,
    },
    {
        "id": "tiff_with_dtype",
        "spec": {
            "driver": "tiff",
            "kvstore": {"driver": "memory"},
            "dtype": "uint8",
        },
        "skip_creation": True,
    },
    {
        "id": "tiff_multipage",
        "spec": {
            "driver": "tiff",
            "kvstore": {"driver": "memory"},
            "path": "multipage.tiff",
            "page": 2,
        },
        "skip_creation": True,
    },
    # With various contexts and options
    {
        "id": "with_cache_pool",
        "spec": {
            "driver": "zarr",
            "kvstore": {"driver": "memory"},
            "context": {"cache_pool": {"total_bytes_limit": 10_000_000}},
        },
        "skip_creation": True,
    },
    {
        "id": "with_creation_flags",
        "spec": {
            "driver": "zarr3",
            "kvstore": {"driver": "memory"},
            "create": True,
            "delete_existing": True,
            "metadata": {
                "shape": [100, 100],
                "data_type": "int16",
            },
        },
    },
    # Adapter drivers (virtual views over a base TensorStore)
    {
        "id": "cast_over_array",
        "spec": {
            "driver": "cast",
            "base": {"driver": "array", "array": [1, 2, 3], "dtype": "int32"},
            "dtype": "float32",
        },
    },
    {
        "id": "downsample_over_array",
        "spec": {
            "driver": "downsample",
            "base": {"driver": "array", "array": [[1, 2], [3, 4]], "dtype": "int32"},
            "downsample_factors": [2, 2],
            "downsample_method": "mean",
        },
    },
    {
        "id": "stack_of_arrays",
        "spec": {
            "driver": "stack",
            "layers": [
                {"driver": "array", "array": [1, 2, 3], "dtype": "int32"},
                {
                    "driver": "array",
                    "array": [4, 5, 6],
                    "dtype": "int32",
                    "transform": {
                        "input_inclusive_min": [3],
                        "output": [{"input_dimension": 0, "offset": -3}],
                    },
                },
            ],
        },
    },
    {
        "id": "json_pointer",
        "spec": {
            "driver": "json",
            "kvstore": "memory://attributes.json",
            "json_pointer": "/a/b",
        },
    },
    # Image drivers (validation-only)
    {
        "id": "png_options",
        "spec": {"driver": "png", "kvstore": "memory://x.png", "compression_level": 3},
        "skip_creation": True,
    },
    {
        "id": "jpeg_quality",
        "spec": {"driver": "jpeg", "kvstore": "memory://x.jpg", "quality": 50},
        "skip_creation": True,
    },
    {
        "id": "webp_lossless",
        "spec": {"driver": "webp", "kvstore": "memory://x.webp", "lossless": False},
        "skip_creation": True,
    },
    {
        "id": "avif_speed",
        "spec": {"driver": "avif", "kvstore": "memory://x.avif", "speed": 8},
        "skip_creation": True,
    },
    {
        "id": "bmp_basic",
        "spec": {"driver": "bmp", "kvstore": "memory://x.bmp"},
        "skip_creation": True,
    },
    # KvStore adapters and remote stores
    {
        "id": "zarr3_over_ocdbt",
        "spec": {
            "driver": "zarr3",
            "kvstore": {
                "driver": "ocdbt",
                "base": "memory://",
                "config": {"max_inline_value_bytes": 100, "manifest_kind": "single"},
            },
            "metadata": {"shape": [4], "data_type": "uint8"},
        },
    },
    {
        "id": "zarr3_over_kvstack",
        "spec": {
            "driver": "zarr3",
            "kvstore": {
                "driver": "kvstack",
                "layers": [
                    {"base": "memory://base/"},
                    {"base": "memory://prefix/", "prefix": "c/", "strip_prefix": 0},
                ],
            },
            "metadata": {"shape": [4], "data_type": "uint8"},
        },
    },
    {
        "id": "zarr3_over_sharding_indexed",
        "spec": {
            "driver": "zarr3",
            "kvstore": {
                "driver": "zarr3_sharding_indexed",
                "base": "memory://",
                "grid_shape": [2],
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "crc32c"},
                ],
            },
            "metadata": {"shape": [4], "data_type": "uint8"},
        },
        # a sharded kvstore only holds chunk keys, so zarr.json cannot be created
        "skip_creation": True,
    },
    {
        "id": "zarr3_over_zip",
        "spec": {
            "driver": "zarr3",
            "kvstore": {"driver": "zip", "base": "memory://a.zip"},
        },
        "skip_creation": True,
    },
    {
        "id": "neuroglancer_sharded",
        "spec": {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "memory"},
            "multiscale_metadata": {
                "type": "image",
                "data_type": "uint8",
                "num_channels": 1,
            },
            "scale_metadata": {
                "size": [64, 64, 64],
                "chunk_size": [16, 16, 16],
                "resolution": [1, 1, 1],
                "encoding": "raw",
                "sharding": {
                    "@type": "neuroglancer_uint64_sharded_v1",
                    "preshift_bits": 1,
                    "hash": "identity",
                    "minishard_bits": 2,
                    "shard_bits": 3,
                    "data_encoding": "gzip",
                    "minishard_index_encoding": "gzip",
                },
            },
        },
    },
    {
        "id": "gcs_kvstore",
        "spec": {
            "driver": "zarr3",
            "kvstore": {"driver": "gcs", "bucket": "my-bucket", "path": "data/"},
        },
        "skip_creation": True,
    },
    {
        "id": "http_kvstore",
        "spec": {
            "driver": "zarr3",
            "kvstore": {
                "driver": "http",
                "base_url": "https://example.com:8000",
                "path": "path/to/data",
                "headers": ["Authorization: Bearer XXX"],
            },
        },
        "skip_creation": True,
    },
    {
        "id": "s3_kvstore_options",
        "spec": {
            "driver": "zarr3",
            "kvstore": {
                "driver": "s3",
                "bucket": "my-bucket",
                "aws_region": "us-east-1",
                "endpoint": "https://s3.example.com",
                "requester_pays": True,
            },
        },
        "skip_creation": True,
    },
    {
        "id": "tsgrpc_kvstore",
        "spec": {
            "driver": "zarr3",
            "kvstore": {"driver": "tsgrpc_kvstore", "address": "localhost:1234"},
        },
        "skip_creation": True,
    },
    {
        "id": "kvstore_pipeline_url",
        "spec": {"driver": "zarr3", "kvstore": "memory://a.zip|zip:"},
        "skip_creation": True,
    },
    # Context resources
    {
        "id": "context_resources",
        "spec": {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": "ctx_test"},
            "context": {
                "cache_pool": {"total_bytes_limit": 10_000_000},
                "cache_pool#remote": {"total_bytes_limit": 100_000_000},
                "data_copy_concurrency": {"limit": 8},
                "file_io_concurrency": {"limit": "shared"},
                "file_io_sync": False,
                "file_io_locking": {"mode": "lockfile"},
                "http_request_concurrency": {"limit": 4},
                "http_request_retries": {"max_retries": 3, "initial_delay": "2s"},
                "aws_credentials": {"type": "anonymous"},
                "gcs_user_project": {"project_id": "my-project"},
            },
            "metadata_cache_pool": "cache_pool#remote",
            "metadata": {"shape": [4], "data_type": "uint8"},
        },
    },
    # Driver aliases and newer options
    {
        "id": "zarr2_driver_alias",
        "spec": {
            "driver": "zarr2",
            "kvstore": {"driver": "memory"},
            "metadata": {"shape": [4], "chunks": [4], "dtype": "<u2"},
        },
    },
    {
        "id": "zarr3_field_and_void",
        "spec": {"driver": "zarr3", "kvstore": "memory://", "open_as_void": True},
        "skip_creation": True,
    },
    # Auto driver examples (validation-only, no actual creation)
    {
        "id": "auto_memory_basic",
        "spec": {"driver": "auto", "kvstore": {"driver": "memory"}},
        "skip_creation": True,
    },
    {
        "id": "auto_file_basic",
        "spec": {"driver": "auto", "kvstore": {"driver": "file", "path": "auto_test"}},
        "skip_creation": True,
    },
    {
        "id": "auto_with_dtype",
        "spec": {"driver": "auto", "kvstore": {"driver": "memory"}, "dtype": "float32"},
        "skip_creation": True,
    },
]


@pytest.mark.parametrize("test_case", ROUND_TRIP_TEST_CASES, ids=lambda x: x["id"])
def test_round_trip_validation(
    test_case: dict, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Test round-trip validation: dict -> our_spec -> tensorstore -> our_spec."""
    spec_dict: dict = test_case["spec"]

    # Use a temporary path for file-based kvstores
    kvstore = spec_dict.get("kvstore", {})
    if isinstance(kvstore, dict) and kvstore.get("driver") == "file":
        tmp_path = tmp_path_factory.mktemp(kvstore["path"])
        spec_dict["kvstore"]["path"] = str(tmp_path)

    # ensure tensorstore recognizes the spec (older releases may lack a feature)
    try:
        ts_spec = ts.Spec(spec_dict)
    except ValueError as e:
        skip_if_older_tensorstore(e)

    # First validate our spec
    our_spec = validate_spec(spec_dict)

    # Validate that we can also validate the tensorstore spec object
    validate_spec(ts_spec)

    our_spec.model_dump(mode="json", exclude_none=True)

    ts_roundtrip = our_spec.to_tensorstore()

    # The round trip should work
    assert ts_roundtrip == ts_spec

    # create an actual tensorstore object to ensure the spec is valid
    if not test_case.get("skip_creation", False):
        ts.open(ts_roundtrip, create=True).result()
        if isinstance(store := getattr(our_spec, "kvstore", None), pts.FileKvStore):
            assert Path(store.path).exists()


def test_example() -> None:
    # from the readme

    spec = pts.Zarr2Spec(
        kvstore=pts.MemoryKvStore(),
        metadata=pts.Zarr2Metadata(
            chunks=[64, 64],
            compressor=pts.Zarr2CompressorBlosc(cname="lz4", clevel=5),
            dtype="<f4",
        ),
    )

    spec.to_tensorstore()


def test_kvstore_string_parsing() -> None:
    """Test kvstore string parsing functionality."""

    # Test file:// URLs
    spec = pts.validate_spec(
        {
            "driver": "zarr",
            "kvstore": "file:///tmp/test",
            "create": True,
            "metadata": {"shape": [10, 10], "dtype": "<f4"},
        }
    )
    assert isinstance(spec.kvstore, pts.FileKvStore)
    assert spec.kvstore.path == "/tmp/test"

    # Test memory:// URLs
    spec = pts.validate_spec(
        {
            "driver": "zarr",
            "kvstore": "memory://",
            "create": True,
            "metadata": {"shape": [10, 10], "dtype": "<f4"},
        }
    )
    assert isinstance(spec.kvstore, pts.MemoryKvStore)

    # Test memory:// URLs with path
    spec = pts.validate_spec(
        {
            "driver": "zarr",
            "kvstore": "memory://test_path",
            "create": True,
            "metadata": {"shape": [10, 10], "dtype": "<f4"},
        }
    )
    assert isinstance(spec.kvstore, pts.MemoryKvStore)
    assert spec.kvstore.path == "test_path"

    # Test s3:// URLs
    spec = pts.validate_spec(
        {
            "driver": "zarr",
            "kvstore": "s3://bucket-name",
            "create": True,
            "metadata": {"shape": [10, 10], "dtype": "<f4"},
        }
    )
    assert isinstance(spec.kvstore, pts.S3KvStore)
    assert spec.kvstore.bucket == "bucket-name"

    # Test s3:// URLs with path
    spec = pts.validate_spec(
        {
            "driver": "zarr",
            "kvstore": "s3://bucket-name/path/to/data",
            "create": True,
            "metadata": {"shape": [10, 10], "dtype": "<f4"},
        }
    )
    assert isinstance(spec.kvstore, pts.S3KvStore)
    assert spec.kvstore.bucket == "bucket-name"
    assert spec.kvstore.path == "path/to/data"

    # gs:// and http(s):// are parsed; anything else is passed through verbatim
    spec = pts.validate_spec({"driver": "zarr3", "kvstore": "gs://bucket-name/p"})
    assert isinstance(spec.kvstore, pts.GCSKvStore)
    assert spec.kvstore.bucket == "bucket-name"
    spec = pts.validate_spec({"driver": "zarr3", "kvstore": "https://x.com/a/b"})
    assert isinstance(spec.kvstore, pts.HTTPKvStore)
    assert spec.kvstore.base_url == "https://x.com"
    assert spec.kvstore.path == "/a/b"  # tensorstore keeps the leading slash

    # paths are percent-decoded, exactly as tensorstore does
    for url in ("file:///data/a%20b", "s3://bucket/a%20b", "memory://a%20b"):
        assert pts.validate_kvstore(url).model_dump() == ts.KvStore.Spec(url).to_json()
    spec = pts.validate_spec({"driver": "zarr3", "kvstore": "memory://a.zip|zip:"})
    assert spec.kvstore == "memory://a.zip|zip:"
    if TS_VERSION is not None and TS_VERSION >= PINNED_VERSION:
        assert spec.to_tensorstore().to_json()["kvstore"]["driver"] == "zip"


# Specs that can be created for real, so that tensorstore's *own* output can be fed
# back through the models. This is the direction users actually rely on, and the one
# that hid two bugs: ChunkLayout demanding a `rank` tensorstore infers, and N5 blosc
# omitting the `blocksize` tensorstore emits.
CREATABLE = [
    {
        "driver": "zarr3",
        "kvstore": "memory://",
        "dtype": "uint16",
        "schema": {"domain": {"shape": [100, 200]}},
    },
    {
        "driver": "zarr",
        "kvstore": "memory://",
        "dtype": "float32",
        "schema": {"domain": {"shape": [50, 60]}},
    },
    {"driver": "n5", "kvstore": "memory://", "dtype": "uint16"},
    {
        "driver": "neuroglancer_precomputed",
        "kvstore": "memory://",
        "multiscale_metadata": {
            "type": "image",
            "data_type": "uint8",
            "num_channels": 1,
        },
        "scale_metadata": {
            "size": [64, 64, 64],
            "chunk_size": [16, 16, 16],
            "resolution": [1, 1, 1],
            "encoding": "raw",
        },
    },
    {
        "driver": "zarr3",
        "kvstore": {"driver": "ocdbt", "base": "memory://"},
        "dtype": "int32",
        "schema": {"domain": {"shape": [10]}},
    },
]


@pytest.fixture(
    params=CREATABLE,
    ids=lambda s: (
        s["driver"] + "-" + s["kvstore"]["driver"]
        if isinstance(s["kvstore"], dict)
        else s["driver"]
    ),
)
def created_store(request: pytest.FixtureRequest) -> ts.TensorStore:
    spec = dict(request.param)
    if spec["driver"] == "n5":
        spec = {**spec, "schema": {"domain": {"shape": [100, 200]}}}
    return ts.open(spec, create=True).result()


def test_created_store_spec_round_trips(created_store: ts.TensorStore) -> None:
    """tensorstore's own spec output validates and round-trips unchanged."""
    for kwargs in ({}, {"minimal_spec": True}):
        raw = created_store.spec(**kwargs).to_json()
        ours = validate_spec(raw)
        assert ours.to_tensorstore() == ts.Spec(raw)


def test_created_store_schema_round_trips(created_store: ts.TensorStore) -> None:
    """`store.schema` / `store.chunk_layout` / `store.codec` validate and round-trip."""
    schema_json = created_store.schema.to_json()
    assert pts.Schema.model_validate(schema_json).model_dump(mode="json") == schema_json

    layout_json = created_store.chunk_layout.to_json()
    assert (
        pts.ChunkLayout.model_validate(layout_json).model_dump(mode="json")
        == layout_json
    )

    if (codec := created_store.codec) is not None:
        codec_json = codec.to_json()
        adapter = TypeAdapter[pts.Codec](pts.Codec)
        assert adapter.validate_python(codec_json).model_dump(mode="json") == codec_json


def test_non_finite_fill_value_survives_json() -> None:
    """NaN must not silently become null: tensorstore reads `null` as a real value."""
    spec = pts.Zarr3Spec(
        kvstore="memory://", schema=pts.Schema(dtype="float32", fill_value=float("nan"))
    )
    assert math.isnan(spec.to_tensorstore().to_json()["schema"]["fill_value"])
    from_json = ts.Spec(json.loads(spec.model_dump_json()))
    assert math.isnan(from_json.to_json()["schema"]["fill_value"])
