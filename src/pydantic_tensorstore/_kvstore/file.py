"""File system key-value store specification."""

from typing import ClassVar, Literal

from pydantic import Field

from pydantic_tensorstore._kvstore.base import BaseKvStore
from pydantic_tensorstore._types import ContextResource


class FileKvStore(BaseKvStore):
    """Read/write access to the local filesystem.

    Stores keys as files in a local or network-mounted file system.
    """

    model_config: ClassVar = {"extra": "forbid"}

    driver: Literal["file"] = "file"

    path: str = Field(description="Path to root directory on local filesystem.")  # pyright: ignore

    file_io_concurrency: ContextResource | None = None
    file_io_sync: ContextResource | None = None
    file_io_mode: ContextResource | None = None
    file_io_locking: ContextResource | None = None
