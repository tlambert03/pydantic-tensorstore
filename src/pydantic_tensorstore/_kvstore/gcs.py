"""Google Cloud Storage key-value store specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._kvstore.base import BaseKvStore
from pydantic_tensorstore._types import ContextResource


class GCSKvStore(BaseKvStore):
    """Read/write access to Google Cloud Storage (GCS)."""

    driver: Literal["gcs"] = "gcs"
    bucket: str = Field(description="Google Cloud Storage bucket to use.")
    gcs_request_concurrency: ContextResource | None = None
    gcs_user_project: ContextResource | None = None
    gcs_request_retries: ContextResource | None = None
