"""Key-value store specifications for TensorStore."""

from typing import Annotated, Any, TypeAlias

from pydantic import BeforeValidator, Field

from .base import BaseKvStore
from .file import FileKvStore
from .memory import MemoryKvStore
from .s3 import S3KvStore

__all__ = [
    "BaseKvStore",
    "FileKvStore",
    "KvStore",
    "MemoryKvStore",
    "S3KvStore",
]


def _str_to_kv_store(value: Any) -> Any:
    """Convert a string to a kvstore specification dictionary."""
    if not isinstance(value, str):
        return value
    if value.startswith("file://"):
        return {"driver": "file", "path": value[len("file://") :]}
    if value.startswith("memory://"):
        store = {"driver": "memory"}
        if value != "memory://":
            store["path"] = value[len("memory://") :]
        return store
    if value.startswith("s3://"):
        store = {"driver": "s3"}
        bucket = value[len("s3://") :]
        bucket_name, *path = bucket.split("/", 1)
        store["bucket"] = bucket_name
        if path:
            store["path"] = path[0]
        return store
    raise ValueError(f"Invalid kvstore string: {value}")


# Simple Union type for all kvstore specs
KvStore: TypeAlias = Annotated[
    FileKvStore | MemoryKvStore | S3KvStore,
    Field(discriminator="driver"),
    BeforeValidator(_str_to_kv_store),
]
