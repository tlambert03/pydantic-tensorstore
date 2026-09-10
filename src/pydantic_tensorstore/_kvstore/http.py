"""HTTP key-value store specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._kvstore.base import BaseKvStore
from pydantic_tensorstore._types import ContextResource


class HTTPKvStore(BaseKvStore):
    """Read-only access to arbitrary HTTP servers."""

    driver: Literal["http"] = "http"
    base_url: str = Field(description="Base URL included in all requests.")
    headers: list[str] | None = Field(
        default=None,
        description='Additional headers, e.g. `["Authorization: Bearer XXX"]`.',
    )
    http_request_concurrency: ContextResource | None = None
    http_request_retries: ContextResource | None = None
