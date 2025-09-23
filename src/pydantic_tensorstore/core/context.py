"""Context models for TensorStore specifications.

Context resources manage shared components like cache pools,
concurrency limits, and network configurations.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from pydantic_tensorstore.types.common import ContextResource, ContextResourceName


class CachePool(ContextResource):
    """Cache pool resource for managing memory usage."""

    total_bytes_limit: Optional[int] = Field(
        default=None,
        description="Total memory limit in bytes",
        gt=0,
    )

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize to JSON, excluding defaults if empty."""
        data = self.model_dump(exclude_defaults=True, **kwargs)
        if not data:
            return "{}"
        return super().model_dump_json(**kwargs)


class DataCopyConcurrency(ContextResource):
    """Concurrency limits for data copy operations."""

    limit: int = Field(
        default=4,
        description="Maximum concurrent data copy operations",
        gt=0,
    )


class FileIOConcurrency(ContextResource):
    """Concurrency limits for file I/O operations."""

    limit: int = Field(
        default=4,
        description="Maximum concurrent file I/O operations",
        gt=0,
    )


class HTTPConcurrency(ContextResource):
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

    Attributes:
        cache_pool: Memory cache configuration
        data_copy_concurrency: Data copy operation limits
        file_io_concurrency: File I/O operation limits
        http_concurrency: HTTP request limits

    Example:
        >>> context = Context(
        ...     cache_pool={"total_bytes_limit": 1000000000},  # 1GB
        ...     data_copy_concurrency={"limit": 8},
        ...     http_concurrency={"limit": 16}
        ... )
    """

    model_config = {"extra": "allow", "validate_assignment": True}

    cache_pool: Optional[Union[CachePool, ContextResourceName, dict[str, Any]]] = Field(
        default=None,
        description="Cache pool resource configuration",
    )

    data_copy_concurrency: Optional[
        Union[DataCopyConcurrency, ContextResourceName, dict[str, Any]]
    ] = Field(
        default=None,
        description="Data copy concurrency limits",
    )

    file_io_concurrency: Optional[
        Union[FileIOConcurrency, ContextResourceName, dict[str, Any]]
    ] = Field(
        default=None,
        description="File I/O concurrency limits",
    )

    http_concurrency: Optional[
        Union[HTTPConcurrency, ContextResourceName, dict[str, Any]]
    ] = Field(
        default=None,
        description="HTTP concurrency limits",
    )

    def get_resource(self, name: str) -> Optional[ContextResource]:
        """Get a context resource by name."""
        resource = getattr(self, name, None)
        if isinstance(resource, ContextResource):
            return resource
        if isinstance(resource, dict):
            # Convert dict to appropriate resource type
            if name == "cache_pool":
                return CachePool.model_validate(resource)
            elif name == "data_copy_concurrency":
                return DataCopyConcurrency.model_validate(resource)
            elif name == "file_io_concurrency":
                return FileIOConcurrency.model_validate(resource)
            elif name == "http_concurrency":
                return HTTPConcurrency.model_validate(resource)
        return None

    def set_resource(self, name: str, resource: Union[ContextResource, dict[str, Any], str]) -> None:
        """Set a context resource."""
        setattr(self, name, resource)

    @classmethod
    def create_default(cls) -> Context:
        """Create a context with sensible defaults."""
        return cls(
            cache_pool=CachePool(),
            data_copy_concurrency=DataCopyConcurrency(),
            file_io_concurrency=FileIOConcurrency(),
            http_concurrency=HTTPConcurrency(),
        )