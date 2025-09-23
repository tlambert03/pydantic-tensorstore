"""Utility functions for working with TensorStore specifications."""

from pydantic_tensorstore.utils.builders import (
    ArraySpecBuilder,
    N5SpecBuilder,
    SpecBuilder,
    ZarrSpecBuilder,
)
from pydantic_tensorstore.utils.conversion import (
    normalize_spec,
    spec_from_dict,
    spec_from_json,
    spec_to_dict,
    spec_to_json,
)
from pydantic_tensorstore.utils.introspection import (
    get_driver_capabilities,
    get_spec_info,
    list_registered_drivers,
)

__all__ = [
    "ArraySpecBuilder",
    "N5SpecBuilder",
    # Builder utilities
    "SpecBuilder",
    "ZarrSpecBuilder",
    "get_driver_capabilities",
    # Introspection utilities
    "get_spec_info",
    "list_registered_drivers",
    "normalize_spec",
    "spec_from_dict",
    "spec_from_json",
    # Conversion utilities
    "spec_to_dict",
    "spec_to_json",
]
