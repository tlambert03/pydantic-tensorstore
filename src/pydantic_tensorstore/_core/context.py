"""Context models for TensorStore specifications.

Context resources manage shared components like cache pools,
concurrency limits, and network configurations.
"""

from typing import Any

from pydantic import BaseModel, Field

from pydantic_tensorstore._types import ContextResourceName


class CachePool(BaseModel):
    """Cache pool resource for managing memory usage."""

    total_bytes_limit: int | None = Field(
        default=None,
        description="Total memory limit in bytes",
        gt=0,
    )


class DataCopyConcurrency(BaseModel):
    """Concurrency limits for data copy operations."""

    limit: int = Field(
        default=4,
        description="Maximum concurrent data copy operations",
        gt=0,
    )


class FileIOConcurrency(BaseModel):
    """Concurrency limits for file I/O operations."""

    limit: int = Field(
        default=4,
        description="Maximum concurrent file I/O operations",
        gt=0,
    )


class HTTPConcurrency(BaseModel):
    """Concurrency limits for HTTP requests."""

    limit: int = Field(
        default=32,
        description="Maximum concurrent HTTP requests",
        gt=0,
    )


class Context(BaseModel):
    """TensorStore context specification.

    Manages shared resources like cache pools, concurrency limits,
    and driver-specific configurations.

    Attributes
    ----------
        cache_pool: Memory cache configuration
        data_copy_concurrency: Data copy operation limits
        file_io_concurrency: File I/O operation limits
        http_concurrency: HTTP request limits

    Example:
        >>> context = Context(
        ...     cache_pool={"total_bytes_limit": 1000000000},  # 1GB
        ...     data_copy_concurrency={"limit": 8},
        ...     http_concurrency={"limit": 16},
        ... )
    """

    model_config = {"extra": "allow", "validate_assignment": True}

    cache_pool: CachePool | ContextResourceName | dict[str, Any] | None = Field(
        default=None,
        description="Cache pool resource configuration",
    )

    data_copy_concurrency: (
        DataCopyConcurrency | ContextResourceName | dict[str, Any] | None
    ) = Field(
        default=None,
        description="Data copy concurrency limits",
    )

    file_io_concurrency: (
        FileIOConcurrency | ContextResourceName | dict[str, Any] | None
    ) = Field(
        default=None,
        description="File I/O concurrency limits",
    )

    http_concurrency: HTTPConcurrency | ContextResourceName | dict[str, Any] | None = (
        Field(
            default=None,
            description="HTTP concurrency limits",
        )
    )
