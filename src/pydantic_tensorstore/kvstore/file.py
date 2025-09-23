"""File system key-value store specification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from pydantic_tensorstore.kvstore.base import BaseKvStoreSpec


class FileKvStoreSpec(BaseKvStoreSpec):
    """File system key-value store specification.

    Stores keys as files in a local or network-mounted file system.

    Attributes:
        driver: Must be "file"
        path: Base path for file storage

    Example:
        >>> kvstore = FileKvStoreSpec(
        ...     driver="file",
        ...     path="/tmp/tensorstore_data/"
        ... )
    """

    model_config = {"extra": "forbid"}

    driver: Literal["file"] = Field(
        default="file",
        description="File system key-value store driver",
    )

    path: str = Field(
        description="Base path for file storage",
    )

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, v: Any) -> str:
        """Validate and normalize the file path."""
        if not isinstance(v, (str, Path)):
            raise ValueError("Path must be a string or Path object")

        path_str = str(v)
        if not path_str:
            raise ValueError("Path cannot be empty")

        # Convert to absolute path and normalize
        try:
            normalized_path = str(Path(path_str).expanduser().resolve())
            return normalized_path
        except Exception as e:
            raise ValueError(f"Invalid path '{path_str}': {e}")

    def get_driver_kind(self) -> str:
        """Get the driver kind."""
        return "kvstore"