"""Key-value store specifications for TensorStore."""

from typing import Annotated, TypeAlias

from pydantic import Field

from pydantic_tensorstore.kvstore.base import BaseKvStoreSpec
from pydantic_tensorstore.kvstore.file import FileKvStoreSpec
from pydantic_tensorstore.kvstore.memory import MemoryKvStoreSpec

__all__ = [
    "BaseKvStoreSpec",
    "FileKvStoreSpec",
    "KvStoreSpec",
    "MemoryKvStoreSpec",
]


# Simple Union type for all kvstore specs
KvStoreSpec: TypeAlias = Annotated[
    FileKvStoreSpec | MemoryKvStoreSpec | MemoryKvStoreSpec,
    Field(discriminator="driver"),
]
