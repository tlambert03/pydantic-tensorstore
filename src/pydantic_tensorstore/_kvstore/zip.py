"""ZIP archive kvstore adapter."""

from __future__ import annotations

from typing import Literal

from pydantic_tensorstore._kvstore.base import _CachedKvStoreAdapter


class ZipKvStore(_CachedKvStoreAdapter):
    """Read-only adapter for the ZIP archive format."""

    driver: Literal["zip"] = "zip"
