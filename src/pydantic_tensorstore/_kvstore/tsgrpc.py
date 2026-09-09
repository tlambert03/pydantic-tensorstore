"""tensorstore gRPC kvstore."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._kvstore.base import BaseKvStore
from pydantic_tensorstore._types import ContextResource


class TsGrpcKvStore(BaseKvStore):
    """Read/write key-value store using the tensorstore-specific gRPC protocol."""

    driver: Literal["tsgrpc_kvstore"] = "tsgrpc_kvstore"
    address: str = Field(description="gRPC service address.")
    timeout: str | None = Field(default=None, description="Request timeout.")
    data_copy_concurrency: ContextResource | None = None
