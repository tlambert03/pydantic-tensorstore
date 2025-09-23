"""Memory key-value store specification."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from pydantic_tensorstore.kvstore.base import BaseKvStoreSpec


class MemoryKvStoreSpec(BaseKvStoreSpec):
    """In-memory key-value store specification.

    Provides a simple in-memory key-value store for testing and temporary data.
    Data is not persisted and is lost when the process ends.

    Attributes
    ----------
        driver: Must be "memory"
        path: Path prefix for keys (optional)

    Example:
        >>> kvstore = MemoryKvStoreSpec(driver="memory")
        >>> kvstore_with_path = MemoryKvStoreSpec(driver="memory", path="my_data/")
    """

    model_config: ClassVar = {"extra": "forbid"}

    driver: Literal["memory"] = Field(
        default="memory",
        description="Memory key-value store driver",
    )
