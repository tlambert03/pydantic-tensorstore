from pathlib import Path

import pytest

from pydantic_tensorstore import validate_spec

try:
    import tensorstore as ts
except ImportError:
    pytest.skip("tensorstore not installed", allow_module_level=True)


def test_from_ts_mem_store() -> None:
    spec_dict = {
        "driver": "zarr",
        "kvstore": {"driver": "memory"},
        "metadata": {
            "chunks": [10, 11],
            "shape": [100, 200],
            "fill_value": None,
            "dtype": "<u2",
            "compressor": None,
            "filters": None,
            "order": "C",
        },
    }
    zarr_mem = ts.open(spec_dict, create=True).result()

    assert validate_spec(zarr_mem)
    assert validate_spec(zarr_mem.spec())
    assert validate_spec(spec_dict)


def test_from_ts_zarr_store(tmp_path: Path) -> None:
    spec_dict = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": str(tmp_path / "zarr_test"),
        },
        "metadata": {
            "shape": [3, 4, 5],
            "data_type": "float32",
            "chunk_key_encoding": {"name": "v2"},
        },
        "create": True,
        "delete_existing": True,
    }
    z3 = ts.open(spec_dict).result()
    assert validate_spec(z3)
    assert validate_spec(spec_dict)
