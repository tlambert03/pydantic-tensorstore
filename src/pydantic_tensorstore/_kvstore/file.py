"""File system key-value store specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._core.base import since
from pydantic_tensorstore._kvstore.base import BaseKvStore
from pydantic_tensorstore._types import ContextResource


class FileKvStore(BaseKvStore):
    """Read/write access to the local filesystem."""

    driver: Literal["file"] = "file"
    path: str = Field(description="Path to root directory on local filesystem.")  # pyright: ignore
    file_io_concurrency: ContextResource | None = None
    file_io_sync: ContextResource | None = None
    file_io_mode: ContextResource | None = Field(
        default=None, json_schema_extra=since("0.1.77")
    )
    file_io_locking: ContextResource | None = None
