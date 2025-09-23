"""Base key-value store specification."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from pydantic import BaseModel, Field


class BaseKvStoreSpec(BaseModel, ABC):
    """Base class for key-value store specifications.

    Key-value stores provide the underlying storage layer for many TensorStore
    drivers, abstracting over local files, cloud storage, databases, etc.

    Attributes
    ----------
        driver: The kvstore driver identifier
        path: Path within the key-value store

    Example:
        >>> # Use concrete implementations like MemoryKvStoreSpec
        >>> from pydantic_tensorstore.kvstore import MemoryKvStoreSpec
        >>> kvstore = MemoryKvStoreSpec(driver="memory")
    """

    model_config: ClassVar = {"extra": "forbid", "validate_assignment": True}

    # driver: DriverName = Field(description="Key-value store driver identifier")

    # path: str = Field(
    #     default="",
    #     description="Path within the key-value store",
    # )
