"""Memory key-value store specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._kvstore.base import BaseKvStore
from pydantic_tensorstore._types import ContextResource


class MemoryKvStore(BaseKvStore):
    """In-memory key-value store. Data is lost when the process ends."""

    driver: Literal["memory"] = "memory"
    memory_key_value_store: ContextResource | None = None
    atomic: bool | None = Field(
        default=None,
        description="Support atomic multi-key transactions. Default: `true`.",
    )
