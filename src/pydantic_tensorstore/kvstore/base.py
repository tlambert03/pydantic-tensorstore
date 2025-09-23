"""Base key-value store specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

from pydantic import BaseModel, Discriminator, Field
from typing_extensions import Annotated

from pydantic_tensorstore.types.common import DriverName


class BaseKvStoreSpec(BaseModel, ABC):
    """Base class for key-value store specifications.

    Key-value stores provide the underlying storage layer for many TensorStore
    drivers, abstracting over local files, cloud storage, databases, etc.

    Attributes:
        driver: The kvstore driver identifier
        path: Path within the key-value store

    Example:
        >>> # Use concrete implementations like MemoryKvStoreSpec
        >>> from pydantic_tensorstore.kvstore import MemoryKvStoreSpec
        >>> kvstore = MemoryKvStoreSpec(driver="memory")
    """

    model_config = {"extra": "forbid", "validate_assignment": True}

    driver: DriverName = Field(description="Key-value store driver identifier")

    path: str = Field(
        default="",
        description="Path within the key-value store",
    )

    @abstractmethod
    def get_driver_kind(self) -> str:
        """Get the driver kind for this kvstore spec."""
        ...


def get_kvstore_discriminator(v: Any) -> str:
    """Discriminator function for kvstore specs."""
    if isinstance(v, dict):
        driver = v.get("driver")
        if isinstance(driver, str):
            return driver
        raise ValueError("Missing or invalid 'driver' field in kvstore")

    if hasattr(v, "driver"):
        return str(v.driver)

    raise ValueError("Cannot determine kvstore driver from input")


# Import kvstore specs for the discriminated union
def _import_kvstore_specs() -> dict[str, type[BaseKvStoreSpec]]:
    """Import all kvstore specs and return a mapping."""
    try:
        from pydantic_tensorstore.kvstore.memory import MemoryKvStoreSpec
        from pydantic_tensorstore.kvstore.file import FileKvStoreSpec

        return {
            "memory": MemoryKvStoreSpec,
            "file": FileKvStoreSpec,
        }
    except ImportError:
        return {}


# Create discriminated union for all kvstore specs
_kvstore_specs = _import_kvstore_specs()

if _kvstore_specs:
    from typing import get_args

    # Create annotated types for each kvstore
    _kvstore_annotations = []
    for driver_name, spec_class in _kvstore_specs.items():
        from pydantic import Tag
        _kvstore_annotations.append(
            Annotated[spec_class, Tag(driver_name)]
        )

    # Create the discriminated union type
    KvStoreSpec = Annotated[
        Union[tuple(_kvstore_annotations)],
        Discriminator(get_kvstore_discriminator),
    ]
else:
    # Fallback if no kvstore specs are available
    KvStoreSpec = BaseKvStoreSpec