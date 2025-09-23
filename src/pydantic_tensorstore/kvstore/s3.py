"""S3 key-value store specification."""

from typing import Annotated, Literal

from pydantic import Field

from pydantic_tensorstore._types import ContextResource
from pydantic_tensorstore.kvstore.base import BaseKvStore


class S3KvStore(BaseKvStore):
    """Read/write access to Amazon S3-compatible object stores."""

    driver: Literal["s3"] = "s3"
    bucket: str

    requester_pays: bool = False
    aws_region: str | None = None
    endpoint: Annotated[str, Field(pattern=r"^https?://")] | None = None
    host_header: str | None = None
    use_conditional_write: bool | None = None
    aws_credentials: ContextResource | None = None
    s3_request_concurrency: ContextResource | None = None
    s3_request_retries: ContextResource | None = None
    experimental_s3_rate_limiter: ContextResource | None = None
    data_copy_concurrency: ContextResource = "data_copy_concurrency"
