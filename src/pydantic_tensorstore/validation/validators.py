"""High-level validation functions for TensorStore specifications."""

from typing import Any

from pydantic import TypeAdapter

from pydantic_tensorstore.core._union import TensorStoreSpec


def validate_spec(spec: Any, strict: bool = True) -> TensorStoreSpec:
    """Validate a TensorStore specification.

    Parameters
    ----------
    spec : dict or TensorStoreSpec
        Specification to validate
    strict : bool, default True
        If True, performs strict validation

    Returns
    -------
    TensorStoreSpec
        Validated specification object
    """
    return TypeAdapter(TensorStoreSpec).validate_python(spec, strict=strict)
