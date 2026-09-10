"""Every data type we accept must be accepted by tensorstore too."""

from __future__ import annotations

import pytest
from conftest import skip_if_older_tensorstore

import pydantic_tensorstore as pts

ts = pytest.importorskip("tensorstore")


def _ts_spec(spec: dict) -> ts.Spec:
    try:
        return ts.Spec(pts.validate_spec(spec).model_dump(mode="json"))
    except ValueError as e:
        skip_if_older_tensorstore(e)


@pytest.mark.parametrize("dtype", list(pts.DataType), ids=str)
def test_top_level_dtype(dtype: pts.DataType) -> None:
    spec = {"driver": "zarr", "kvstore": "memory://", "dtype": dtype}
    assert _ts_spec(spec).dtype.name == dtype


@pytest.mark.parametrize("dtype", sorted(pts.VALID_ZARR3_DTYPES), ids=str)
def test_zarr3_dtype(dtype: pts.DataType) -> None:
    spec = {"driver": "zarr3", "kvstore": "memory://", "metadata": {"data_type": dtype}}
    _ts_spec(spec)


def test_zarr3_raw_dtype() -> None:
    spec = {"driver": "zarr3", "kvstore": "memory://", "metadata": {"data_type": "r8"}}
    assert _ts_spec(spec).dtype == ts.byte


@pytest.mark.parametrize("dtype", sorted(pts.VALID_N5_DTYPES), ids=str)
def test_n5_dtype(dtype: pts.DataType) -> None:
    spec = {"driver": "n5", "kvstore": "memory://", "metadata": {"dataType": dtype}}
    _ts_spec(spec)


@pytest.mark.parametrize("dtype", sorted(pts.VALID_NEUROGLANCER_DTYPES), ids=str)
def test_neuroglancer_dtype(dtype: pts.DataType) -> None:
    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": "memory://",
        "multiscale_metadata": {"data_type": dtype},
    }
    _ts_spec(spec)
