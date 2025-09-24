from importlib.metadata import version
from pathlib import Path

import pytest

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
]

ts_version = tuple(int(x) for x in version("tensorstore").split(".")[:3])
if ts_version >= (0, 1, 76):
    ROUND_TRIP_TEST_CASES += [
        # Auto driver examples (validation-only, no actual creation)
        {
            "id": "auto_memory_basic",
            "spec": {
                "driver": "auto",
                "kvstore": {"driver": "memory"},
            },
            "skip_creation": True,
        },
        {
            "id": "auto_file_basic",
            "spec": {
                "driver": "auto",
                "kvstore": {"driver": "file", "path": "auto_test"},
            },
            "skip_creation": True,
        },
        {
            "id": "auto_with_dtype",
            "spec": {
                "driver": "auto",
                "kvstore": {"driver": "memory"},
                "dtype": "float32",
            },
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
    if kvstore.get("driver") == "file":
        tmp_path = tmp_path_factory.mktemp(kvstore["path"])
        spec_dict["kvstore"]["path"] = str(tmp_path)

    # ensure tensorstore recognizes the spec
    ts_spec = ts.Spec(spec_dict)

    # First validate our spec
    our_spec = validate_spec(spec_dict)

    # Validate that we can also validate the tensorstore spec object
    validate_spec(ts_spec)

    our_spec.model_dump(mode="json", exclude_none=True)

    ts_roundtrip = our_spec.to_tensorstore()

    # The round trip should work
    assert ts_roundtrip == ts_spec

    # create an actual tensorstore object to ensure the spec is valid
    if spec_dict.get("create") is not False and not test_case.get(
        "skip_creation", False
    ):
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
