"""High-level validation functions for TensorStore specifications."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import TypeAdapter

if TYPE_CHECKING:
    from pydantic_tensorstore import KvStore, TensorStoreSpec
T = TypeVar("T")


@cache
def _spec_adapter() -> TypeAdapter[TensorStoreSpec]:
    from pydantic_tensorstore import TensorStoreSpec

    return TypeAdapter(TensorStoreSpec)


@cache
def _kvstore_adapter() -> TypeAdapter[KvStore]:
    from pydantic_tensorstore import KvStore

    return TypeAdapter(KvStore)


def _validate(adapter: TypeAdapter[T], obj: Any, strict: bool) -> T:
    if isinstance(obj, str | bytes | bytearray) and adapter is _spec_adapter():
        return adapter.validate_json(obj, strict=strict)
    return adapter.validate_python(obj, strict=strict)


def validate_spec(spec: Any, strict: bool = False) -> TensorStoreSpec:
    """Validate a TensorStore specification.

    Parameters
    ----------
    spec : dict, str, bytes, tensorstore.Spec, tensorstore.TensorStore, TensorStoreSpec
        Specification to validate. Strings/bytes are parsed as JSON.
    strict : bool, default False
        If True, performs strict validation

    Returns
    -------
    TensorStoreSpec
        Validated specification object
    """
    return _validate(_spec_adapter(), spec, strict)


def validate_kvstore(kvstore: Any, strict: bool = False) -> KvStore:
    """Validate a key-value store specification (dict, URL string, or model)."""
    return _validate(_kvstore_adapter(), kvstore, strict)
