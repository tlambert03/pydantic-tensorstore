"""Memory key-value store specification."""

from typing import Literal

from pydantic_tensorstore._kvstore.base import BaseKvStore
from pydantic_tensorstore._types import ContextResource


class MemoryKvStore(BaseKvStore):
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

    driver: Literal["memory"] = "memory"
    memory_key_value_store: ContextResource | None = None
    atomic: bool = True
