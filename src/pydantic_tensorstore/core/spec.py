"""Base TensorStore specification models.

Defines the main TensorStoreSpec class and driver registry system.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from pydantic import BaseModel, Field

from pydantic_tensorstore.core.context import Context  # noqa: TC001
from pydantic_tensorstore.core.schema import Schema  # noqa: TC001
from pydantic_tensorstore.core.transform import IndexTransform  # noqa: TC001
from pydantic_tensorstore.types.common import JsonObject  # noqa: TC001


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

    model_config: ClassVar = {"extra": "forbid", "validate_assignment": True}

    # driver: DriverName = Field(description="TensorStore driver identifier")

    context: Context | JsonObject | None = Field(
        default=None,
        description="Context resource configuration",
    )

    schema: Schema | JsonObject | None = Field(  # type: ignore[assignment]
        default=None,
        description="Schema constraints",
    )

    transform: IndexTransform | JsonObject | None = Field(
        default=None,
        description="Index transform",
    )
