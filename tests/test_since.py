"""`since` markers: which tensorstore release each newer feature needs."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import PINNED_VERSION, TS_VERSION
from pydantic import BaseModel

import pydantic_tensorstore as pts
from pydantic_tensorstore._core.base import parse_version

MEM = "memory://"

# One spec per `since` entry: (spec, version it must report).  Together with the
# tensorstore version matrix in CI, this keeps the table honest: an older
# tensorstore must reject the spec, a newer one must accept it.
SAMPLES: list[tuple[dict[str, Any], str]] = [
    ({"driver": "zarr3", "kvstore": MEM}, pts.MIN_TENSORSTORE_VERSION),
    (
        {
            "driver": "zarr3",
            "kvstore": MEM,
            "context": {"aws_credentials": {"type": "anonymous"}},
        },
        "0.1.72",
    ),
    (
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "s3",
                "bucket": "my-bucket",
                "use_conditional_write": True,
            },
        },
        "0.1.74",
    ),
    ({"driver": "zarr2", "kvstore": MEM}, "0.1.75"),
    ({"driver": "zarr", "kvstore": MEM, "dtype": "int2"}, "0.1.75"),
    (
        {"driver": "zarr", "kvstore": MEM, "metadata": {"dtype": "float8_e3m4"}},
        "0.1.75",
    ),
    ({"driver": "zarr", "kvstore": MEM, "schema": {"dtype": "int2"}}, "0.1.75"),
    (
        {
            "driver": "cast",
            "base": {"driver": "array", "array": [1], "dtype": "int32"},
            "dtype": "int2",
        },
        "0.1.75",
    ),
    ({"driver": "auto", "kvstore": MEM}, "0.1.76"),
    (
        {
            "driver": "zarr3",
            "kvstore": MEM,
            "context": {"file_io_mode": {"mode": "memmap"}},
        },
        "0.1.77",
    ),
    (
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": "/tmp/x",
                "file_io_mode": "file_io_mode",
            },
        },
        "0.1.77",
    ),
    (
        {"driver": "zarr3", "kvstore": MEM, "metadata": {"data_type": "float4_e2m1fn"}},
        "0.1.80",
    ),
    ({"driver": "zarr", "kvstore": MEM, "open_as_void": True}, "0.1.81"),
    ({"driver": "zarr3", "kvstore": MEM, "field": "a"}, "0.1.85"),
    ({"driver": "zarr3", "kvstore": MEM, "open_as_void": True}, "0.1.85"),
    ({"driver": "zarr3", "kvstore": MEM, "dtype": "float8_e8m0fnu"}, "0.1.85"),
    ({"driver": "zarr3", "kvstore": MEM, "metadata": {"data_type": "r8"}}, "0.1.85"),
]


def _all_since_markers() -> list[tuple[str, str, str | dict[str, str]]]:
    out = []
    for name in pts.__all__:
        obj = getattr(pts, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            for fname, field in obj.model_fields.items():
                extra = field.json_schema_extra
                if isinstance(extra, dict) and "since" in extra:
                    out.append((name, fname, extra["since"]))
    return out


def test_markers_are_within_supported_range() -> None:
    lo, hi = parse_version(pts.MIN_TENSORSTORE_VERSION), PINNED_VERSION
    markers = _all_since_markers()
    assert markers, "no since markers found"
    for model, field, since in markers:
        versions = [since] if isinstance(since, str) else list(since.values())
        for v in versions:
            assert lo < parse_version(v) <= hi, f"{model}.{field}: since={v}"


@pytest.mark.parametrize(
    ("spec", "expected"), SAMPLES, ids=lambda x: x if isinstance(x, str) else ""
)
def test_required_version(spec: dict[str, Any], expected: str) -> None:
    model = pts.validate_spec(spec)
    assert model.required_tensorstore_version() == expected


@pytest.mark.parametrize(
    ("spec", "expected"), SAMPLES, ids=lambda x: x if isinstance(x, str) else ""
)
def test_table_matches_installed_tensorstore(
    spec: dict[str, Any], expected: str
) -> None:
    """The installed tensorstore accepts the spec iff it is at least `expected`."""
    ts = pytest.importorskip("tensorstore")
    assert TS_VERSION is not None
    model = pts.validate_spec(spec)
    if TS_VERSION >= parse_version(expected):
        ts.Spec(model.model_dump(mode="json"))
        model.to_tensorstore()
    else:
        with pytest.raises(ValueError):
            ts.Spec(model.model_dump(mode="json"))
        with pytest.raises(
            pts.UnsupportedTensorStoreVersionError, match="requires tensorstore"
        ):
            model.to_tensorstore()
        # the check can be bypassed, in which case tensorstore's own error surfaces
        with pytest.raises(ValueError):
            model.to_tensorstore(check_version=False)


def test_requirements_paths() -> None:
    spec = pts.validate_spec(
        {
            "driver": "cast",
            "base": {"driver": "zarr3", "kvstore": MEM, "open_as_void": True},
            "dtype": "int2",
        }
    )
    assert spec.version_requirements() == [
        ("dtype=int2", "0.1.75"),
        ("base.open_as_void", "0.1.85"),
    ]
    assert spec.required_tensorstore_version() == "0.1.85"


def test_check_without_tensorstore_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_tensorstore._core import base

    monkeypatch.setattr(base, "installed_tensorstore_version", lambda: None)
    pts.Zarr3Spec(kvstore=MEM, open_as_void=True).check_tensorstore_version()
