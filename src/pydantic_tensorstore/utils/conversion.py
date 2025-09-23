"""Conversion utilities for TensorStore specifications."""

from __future__ import annotations

import json
from typing import Any

from pydantic_tensorstore.core.spec import TensorStoreSpec
from pydantic_tensorstore.validation.validators import validate_spec


def spec_to_dict(
    spec: TensorStoreSpec,
    exclude_unset: bool = True,
    exclude_none: bool = True,
) -> dict[str, Any]:
    """Convert a TensorStore spec to a dictionary.

    Parameters
    ----------
    spec : TensorStoreSpec
        The spec to convert
    exclude_unset : bool, default True
        Whether to exclude fields that weren't explicitly set
    exclude_none : bool, default True
        Whether to exclude fields with None values

    Returns
    -------
    dict
        Dictionary representation of the spec
    """
    return spec.model_dump(
        exclude_unset=exclude_unset,
        exclude_none=exclude_none,
        mode="json",
    )


def spec_from_dict(
    spec_dict: dict[str, Any],
    strict: bool = True,
    validate: bool = True,
) -> TensorStoreSpec:
    """Create a TensorStore spec from a dictionary.

    Parameters
    ----------
    spec_dict : dict
        Dictionary representation of the spec
    strict : bool, default True
        Whether to perform strict validation
    validate : bool, default True
        Whether to validate the spec

    Returns
    -------
    TensorStoreSpec
        Parsed and validated spec object
    """
    if validate:
        return validate_spec(spec_dict, strict=strict)
    else:
        return TensorStoreSpec.model_validate(spec_dict)


def spec_to_json(
    spec: TensorStoreSpec,
    exclude_unset: bool = True,
    exclude_none: bool = True,
    indent: int | str | None = None,
) -> str:
    """Convert a TensorStore spec to JSON string.

    Parameters
    ----------
    spec : TensorStoreSpec
        The spec to convert
    exclude_unset : bool, default True
        Whether to exclude fields that weren't explicitly set
    exclude_none : bool, default True
        Whether to exclude fields with None values
    indent : int, str, or None, default None
        JSON indentation (passed to json.dumps)

    Returns
    -------
    str
        JSON string representation of the spec
    """
    spec_dict = spec_to_dict(
        spec, exclude_unset=exclude_unset, exclude_none=exclude_none
    )
    return json.dumps(spec_dict, indent=indent)


def spec_from_json(
    json_str: str,
    strict: bool = True,
    validate: bool = True,
) -> TensorStoreSpec:
    """Create a TensorStore spec from a JSON string.

    Parameters
    ----------
    json_str : str
        JSON string representation of the spec
    strict : bool, default True
        Whether to perform strict validation
    validate : bool, default True
        Whether to validate the spec

    Returns
    -------
    TensorStoreSpec
        Parsed and validated spec object
    """
    spec_dict = json.loads(json_str)
    return spec_from_dict(spec_dict, strict=strict, validate=validate)


def normalize_spec(
    spec: dict[str, Any] | TensorStoreSpec,
    exclude_defaults: bool = True,
) -> dict[str, Any]:
    """Normalize a spec to a consistent dictionary format.

    Useful for comparing specs, serialization, or preparing specs
    for use with the actual TensorStore library.

    Parameters
    ----------
    spec : dict or TensorStoreSpec
        The spec to normalize
    exclude_defaults : bool, default True
        Whether to exclude default values

    Returns
    -------
    dict
        Normalized dictionary representation
    """
    if isinstance(spec, dict):
        # Parse to validate and normalize
        parsed_spec = spec_from_dict(spec)
        return spec_to_dict(parsed_spec, exclude_unset=exclude_defaults)
    else:
        return spec_to_dict(spec, exclude_unset=exclude_defaults)


def merge_specs(
    base_spec: dict[str, Any] | TensorStoreSpec,
    override_spec: dict[str, Any] | TensorStoreSpec,
) -> TensorStoreSpec:
    """Merge two specs, with override_spec taking precedence.

    Parameters
    ----------
    base_spec : dict or TensorStoreSpec
        Base specification
    override_spec : dict or TensorStoreSpec
        Override specification (takes precedence)

    Returns
    -------
    TensorStoreSpec
        Merged specification
    """
    base_dict = normalize_spec(base_spec)
    override_dict = normalize_spec(override_spec)

    # Simple merge - override dict takes precedence
    merged_dict = {**base_dict, **override_dict}

    # Handle nested dicts like schema, context, etc.
    for key in ["schema", "context", "metadata"]:
        if key in base_dict and key in override_dict:
            if isinstance(base_dict[key], dict) and isinstance(
                override_dict[key], dict
            ):
                merged_dict[key] = {**base_dict[key], **override_dict[key]}

    return spec_from_dict(merged_dict)


def compare_specs(
    spec1: dict[str, Any] | TensorStoreSpec,
    spec2: dict[str, Any] | TensorStoreSpec,
    ignore_fields: list[str] | None = None,
) -> bool:
    """Compare two specs for equality.

    Parameters
    ----------
    spec1, spec2 : dict or TensorStoreSpec
        Specs to compare
    ignore_fields : list of str, optional
        Fields to ignore when comparing

    Returns
    -------
    bool
        True if specs are equivalent
    """
    dict1 = normalize_spec(spec1)
    dict2 = normalize_spec(spec2)

    if ignore_fields:
        for field in ignore_fields:
            dict1.pop(field, None)
            dict2.pop(field, None)

    return dict1 == dict2
