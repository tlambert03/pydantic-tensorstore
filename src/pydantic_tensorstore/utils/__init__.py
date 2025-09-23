"""Utility functions for working with TensorStore specifications."""

from pydantic_tensorstore.utils.conversion import (
    spec_to_dict,
    spec_from_dict,
    spec_to_json,
    spec_from_json,
    normalize_spec,
)
from pydantic_tensorstore.utils.builders import (
    SpecBuilder,
    ArraySpecBuilder,
    ZarrSpecBuilder,
    N5SpecBuilder,
)
from pydantic_tensorstore.utils.introspection import (
    get_spec_info,
    get_driver_capabilities,
    list_registered_drivers,
)

__all__ = [
    # Conversion utilities
    "spec_to_dict",
    "spec_from_dict",
    "spec_to_json",
    "spec_from_json",
    "normalize_spec",
    # Builder utilities
    "SpecBuilder",
    "ArraySpecBuilder",
    "ZarrSpecBuilder",
    "N5SpecBuilder",
    # Introspection utilities
    "get_spec_info",
    "get_driver_capabilities",
    "list_registered_drivers",
]