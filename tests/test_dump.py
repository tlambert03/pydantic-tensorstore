"""Serialization semantics: emit only what was set; explicit None survives."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import pydantic_tensorstore as pts


def test_minimal_spec_dumps_only_set_fields() -> None:
    spec = pts.Zarr3Spec(kvstore="memory://")
    assert spec.model_dump() == {"driver": "zarr3", "kvstore": {"driver": "memory"}}
    assert spec.model_dump_json() == '{"kvstore":{"driver":"memory"},"driver":"zarr3"}'


def test_explicit_none_is_preserved() -> None:
    spec = pts.Zarr2Spec(kvstore="memory://", metadata={"compressor": None})
    assert spec.model_dump()["metadata"] == {"compressor": None}
    spec.metadata.fill_value = None  # type: ignore[union-attr]
    assert spec.model_dump()["metadata"] == {"compressor": None, "fill_value": None}


def test_discriminators_are_always_emitted() -> None:
    assert pts.Zarr2CompressorBlosc().model_dump() == {"id": "blosc"}
    assert pts.Zarr3CodecBytes().model_dump() == {"name": "bytes"}
    sharding = pts.NeuroglancerShardingSpec(
        preshift_bits=1, hash="identity", minishard_bits=1, shard_bits=1
    )
    assert sharding.model_dump()["@type"] == "neuroglancer_uint64_sharded_v1"


def test_alias_round_trip() -> None:
    spec = pts.ArraySpec(array=[1], dtype="int32", schema={"rank": 1})
    assert spec.schema_ is not None
    assert spec.model_dump()["schema"] == {"rank": 1}
    assert pts.ArraySpec(array=[1], dtype="int32", schema_={"rank": 1}) == spec


def test_unit_serializes_to_canonical_array() -> None:
    schema = pts.Schema(dimension_units=["4nm", None, 5, ["2", "s"]])
    assert schema.model_dump()["dimension_units"] == [
        [4.0, "nm"],
        None,
        [5.0, ""],
        [2.0, "s"],
    ]


def test_array_keeps_integer_dtype() -> None:
    spec = pts.validate_spec({"driver": "array", "array": [1, 2], "dtype": "int32"})
    assert spec.model_dump()["array"] == [1, 2]
    assert spec.array.dtype == "int32"  # type: ignore[union-attr]
    strings = pts.validate_spec({"driver": "array", "array": ["a"], "dtype": "string"})
    assert strings.model_dump()["array"] == ["a"]


def test_validate_kvstore_url_passthrough() -> None:
    assert isinstance(pts.validate_kvstore("gs://bucket/path"), pts.GCSKvStore)
    assert isinstance(pts.validate_kvstore("https://x.com/a"), pts.HTTPKvStore)
    assert pts.validate_kvstore("memory://a.zip|zip:") == "memory://a.zip|zip:"
    spec = pts.validate_spec({"driver": "zarr3", "kvstore": "memory://a.zip|zip:"})
    assert spec.model_dump()["kvstore"] == "memory://a.zip|zip:"


def test_repr_shows_only_what_was_set() -> None:
    assert repr(pts.Zarr3Spec(kvstore="memory://")) == (
        "Zarr3Spec(kvstore=MemoryKvStore(driver='memory'), driver='zarr3')"
    )
    # an explicit None is meaningful, so it stays visible
    spec = pts.Zarr2Spec(kvstore="memory://", metadata={"compressor": None})
    assert "compressor=None" in repr(spec)


def test_validation_errors_name_the_type() -> None:
    with pytest.raises(ValidationError, match="validation error for TensorStoreSpec"):
        pts.validate_spec({"driver": "zarr3", "kvstore": "memory://", "nope": 1})
    with pytest.raises(ValidationError, match="validation errors for KvStore"):
        pts.validate_kvstore({"driver": "nope"})


def test_array_rejects_lossy_dtype() -> None:
    with pytest.raises(ValidationError, match="cannot be represented"):
        pts.validate_spec({"driver": "array", "array": [1.7], "dtype": "int32"})
    # exact values and non-finite floats are fine
    assert pts.validate_spec(
        {"driver": "array", "array": [1.0], "dtype": "int32"}
    ).model_dump()["array"] == [1]
    assert pts.ArraySpec(array=[float("nan")], dtype="float32").model_dump_json()
