"""S3 key-value store specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._core.base import since
from pydantic_tensorstore._kvstore.base import BaseKvStore
from pydantic_tensorstore._types import ContextResource


class S3KvStore(BaseKvStore):
    """Read/write access to Amazon S3-compatible object stores."""

    driver: Literal["s3"] = "s3"
    bucket: str = Field(description="AWS S3 bucket name.")
    requester_pays: bool | None = Field(
        default=None, description="Permit requester-pays buckets. Default: `false`."
    )
    aws_region: str | None = None
    endpoint: str | None = Field(
        default=None, description="S3 server endpoint, e.g. `https://s3.example.com`."
    )
    host_header: str | None = None
    use_conditional_write: bool | None = Field(
        default=None, json_schema_extra=since("0.1.74")
    )
    aws_credentials: ContextResource | None = None
    s3_request_concurrency: ContextResource | None = None
    s3_request_retries: ContextResource | None = None
    experimental_s3_rate_limiter: ContextResource | None = None
    data_copy_concurrency: ContextResource | None = None
