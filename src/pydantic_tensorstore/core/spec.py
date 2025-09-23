"""Base TensorStore specification models.

Defines the main TensorStoreSpec class and driver registry system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, Discriminator, Field, Tag, field_validator

if TYPE_CHECKING:
    from pydantic_tensorstore.core.context import Context
    from pydantic_tensorstore.core.schema import Schema
    from pydantic_tensorstore.core.transform import IndexTransform
    from pydantic_tensorstore.types.common import DriverName, JsonObject


class BaseDriverSpec(BaseModel, ABC):
    """Base class for all driver-specific specifications.

    Each TensorStore driver implements its own spec by inheriting from this class.
    The driver field is used for discriminated union dispatch.

    Attributes
    ----------
        driver: The driver identifier (required for all specs)
        context: Context resource configuration
        schema: Schema constraints and metadata
        transform: Index transformation to apply

    Example:
        >>> # This is an abstract base - use concrete driver specs instead
        >>> from pydantic_tensorstore.drivers import ArraySpec
        >>> spec = ArraySpec(driver="array", array=[[1, 2], [3, 4]], dtype="int32")
    """

    model_config = {"extra": "forbid", "validate_assignment": True}

    driver: DriverName = Field(description="TensorStore driver identifier")

    context: Context | JsonObject | None = Field(
        default=None,
        description="Context resource configuration",
    )

    schema: Schema | JsonObject | None = Field(
        default=None,
        description="Schema constraints",
    )

    transform: IndexTransform | JsonObject | None = Field(
        default=None,
        description="Index transform",
    )

    @abstractmethod
    def get_driver_kind(self) -> str:
        """Get the driver kind for this spec."""
        ...

    @field_validator("driver", mode="before")
    @classmethod
    def validate_driver_name(cls, v: Any) -> str:
        """Validate the driver name."""
        if not isinstance(v, str):
            raise ValueError("Driver must be a string")
        if not v:
            raise ValueError("Driver cannot be empty")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        # Validate that the driver matches the expected driver for this spec
        expected_driver = getattr(self, "_expected_driver", None)
        if expected_driver and self.driver != expected_driver:
            raise ValueError(
                f"Expected driver '{expected_driver}' but got '{self.driver}'"
            )


def get_driver_discriminator(v: Any) -> str:
    """Discriminator function for driver specs.

    Determines the appropriate spec class based on the driver field.
    """
    if isinstance(v, dict):
        driver = v.get("driver")
        if isinstance(driver, str):
            return driver
        raise ValueError("Missing or invalid 'driver' field")

    if hasattr(v, "driver"):
        return str(v.driver)

    raise ValueError("Cannot determine driver from input")


# Import driver specs for the discriminated union
# Note: These imports are placed here to avoid circular imports
def _import_driver_specs() -> dict[str, type[BaseDriverSpec]]:
    """Import all driver specs and return a mapping."""
    try:
        from pydantic_tensorstore.drivers.array import ArraySpec
        from pydantic_tensorstore.drivers.n5 import N5Spec
        from pydantic_tensorstore.drivers.neuroglancer_precomputed import (
            NeuroglancerPrecomputedSpec,
        )
        from pydantic_tensorstore.drivers.zarr import ZarrSpec
        from pydantic_tensorstore.drivers.zarr3 import Zarr3Spec

        return {
            "array": ArraySpec,
            "zarr": ZarrSpec,
            "zarr3": Zarr3Spec,
            "n5": N5Spec,
            "neuroglancer_precomputed": NeuroglancerPrecomputedSpec,
        }
    except ImportError:
        # Return empty dict if drivers not available
        return {}


# Create discriminated union for all driver specs
_driver_specs = _import_driver_specs()

# Create annotated types for each driver
_driver_annotations = []
for driver_name, spec_class in _driver_specs.items():
    _driver_annotations.append(Annotated[spec_class, Tag(driver_name)])

# Create the discriminated union type
if _driver_annotations:
    TensorStoreSpec = Annotated[
        tuple(_driver_annotations),
        Discriminator(get_driver_discriminator),
    ]
else:
    # Fallback if no drivers are available
    TensorStoreSpec = BaseDriverSpec


class SpecValidator:
    """Utilities for validating TensorStore specifications."""

    @staticmethod
    def validate_spec_dict(spec_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate a spec dictionary before parsing.

        This handles complex validation logic that's difficult to express
        in Pydantic models, such as cross-field validation and fallback logic.

        Args:
            spec_dict: Raw specification dictionary

        Returns
        -------
            Validated and potentially modified specification dictionary

        Raises
        ------
            ValueError: If the specification is invalid
        """
        if not isinstance(spec_dict, dict):
            raise ValueError("Spec must be a dictionary")

        # Check for required driver field
        if "driver" not in spec_dict:
            raise ValueError("Missing required 'driver' field")

        driver = spec_dict["driver"]
        if not isinstance(driver, str):
            raise ValueError("Driver must be a string")

        # Validate driver is registered
        if driver not in _driver_specs:
            # Try to handle as kvstore spec fallback
            try:
                return SpecValidator._try_kvstore_fallback(spec_dict)
            except Exception:
                available_drivers = list(_driver_specs.keys())
                raise ValueError(
                    f"Unknown driver '{driver}'. Available drivers: {available_drivers}"
                )

        # Driver-specific validation
        return SpecValidator._validate_driver_specific(spec_dict)

    @staticmethod
    def _try_kvstore_fallback(spec_dict: dict[str, Any]) -> dict[str, Any]:
        """Try to parse as a kvstore specification."""
        # This would implement the fallback logic from TensorStore C++
        # For now, just raise an error
        raise ValueError("KvStore fallback not implemented yet")

    @staticmethod
    def _validate_driver_specific(spec_dict: dict[str, Any]) -> dict[str, Any]:
        """Perform driver-specific validation."""
        driver = spec_dict["driver"]

        # Add driver-specific validation logic here
        # For example, ensure required fields are present

        if driver == "array":
            if "array" not in spec_dict:
                raise ValueError("Array driver requires 'array' field")
            if "dtype" not in spec_dict:
                raise ValueError("Array driver requires 'dtype' field")

        elif driver == "zarr":
            if "kvstore" not in spec_dict:
                raise ValueError("Zarr driver requires 'kvstore' field")

        # Add more driver-specific validation as needed

        return spec_dict

    @staticmethod
    def get_registered_drivers() -> list[str]:
        """Get list of registered driver names."""
        return list(_driver_specs.keys())

    @staticmethod
    def is_driver_registered(driver: str) -> bool:
        """Check if a driver is registered."""
        return driver in _driver_specs
