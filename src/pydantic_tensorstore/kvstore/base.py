"""Base key-value store specification."""

from typing import ClassVar

from pydantic import BaseModel, Field

from pydantic_tensorstore._types import ContextResource


class BaseKvStore(BaseModel):
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

    # driver: str

    path: str | None = Field(
        default=None,
        description=(
            "Key prefix within the key-value store. If the prefix is intended "
            "to correspond to a Unix-style directory path, it should end with '/'."
        ),
    )

    context: dict[str, ContextResource] | None = Field(
        default=None,
        description=(
            "Specifies context resources that augment/override the parent context."
        ),
    )
