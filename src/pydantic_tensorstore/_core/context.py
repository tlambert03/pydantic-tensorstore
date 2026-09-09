"""Context models for TensorStore specifications.

Context resources manage shared components like cache pools,
concurrency limits, and network configurations.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt

from pydantic_tensorstore._core.base import TensorStoreModel, since

ConcurrencyLimit: TypeAlias = PositiveInt | Literal["shared"]


class CachePool(TensorStoreModel):
    """Cache pool resource for managing memory usage."""

    total_bytes_limit: NonNegativeInt | None = Field(
        default=None,
        description="Soft memory limit in bytes. Default: `0` (no caching).",
    )


class DataCopyConcurrency(TensorStoreModel):
    """Concurrency limits for data copying/encoding/decoding."""

    limit: ConcurrencyLimit | None = Field(
        default=None,
        description='Maximum concurrent operations. Default: `"shared"`.',
    )


class FileIOConcurrency(TensorStoreModel):
    """Concurrency limits for file I/O operations."""

    limit: ConcurrencyLimit | None = Field(
        default=None,
        description='Maximum concurrent operations. Default: `"shared"`.',
    )


class FileIOMode(TensorStoreModel):
    """File I/O mode for the `file` kvstore."""

    mode: Literal["default", "memmap", "direct"] | None = Field(
        default=None, description='I/O mode. Default: `"default"`.'
    )


class FileIOLocking(TensorStoreModel):
    """Locking mode for the `file` kvstore."""

    mode: Literal["os", "lockfile", "none", "non_atomic"] | None = Field(
        default=None, description='Locking mode. Default: `"os"`.'
    )
    acquire_timeout: str | None = Field(
        default=None, description='Lock acquisition timeout. Default: `"60s"`.'
    )


class HTTPRequestConcurrency(TensorStoreModel):
    """Concurrency limits for HTTP requests."""

    limit: ConcurrencyLimit | None = Field(
        default=None,
        description='Maximum concurrent requests. Default: `"shared"`.',
    )


class RequestRetries(TensorStoreModel):
    """Retry parameters for HTTP/GCS/S3 requests."""

    max_retries: PositiveInt | None = Field(
        default=None, description="Maximum number of attempts. Default: `32`."
    )
    initial_delay: str | None = Field(
        default=None, description='Initial backoff delay. Default: `"1s"`.'
    )
    max_delay: str | None = Field(
        default=None, description='Maximum backoff delay. Default: `"32s"`.'
    )


class RateLimiter(TensorStoreModel):
    """Experimental rate limiter for GCS/S3 reads and writes."""

    read_rate: float | None = Field(default=None, description="Reads per second.")
    write_rate: float | None = Field(default=None, description="Writes per second.")
    doubling_time: str | None = Field(
        default=None, description='Rate doubling time. Default: `"0"`.'
    )


class GCSUserProject(TensorStoreModel):
    """Project to bill for GCS requests."""

    project_id: str | None = Field(default=None, description="GCP project id.")


class MemoryKeyValueStore(TensorStoreModel):
    """Backing store for the `memory` kvstore. No options."""


class OcdbtCoordinator(TensorStoreModel):
    """Distributed coordination server for OCDBT."""

    address: str | None = Field(
        default=None, description="Address of gRPC coordinator server."
    )
    lease_duration: str | None = Field(
        default=None, description='Lease duration. Default: `"10s"`.'
    )


class _AwsCredentials(TensorStoreModel):
    """Base for AWS credential providers."""


class AwsCredentialsAnonymous(_AwsCredentials):
    """Anonymous (unsigned) requests."""

    type: Literal["anonymous"] = "anonymous"


class AwsCredentialsEnvironment(_AwsCredentials):
    """Credentials from `AWS_*` environment variables."""

    type: Literal["environment"] = "environment"


class AwsCredentialsImds(_AwsCredentials):
    """Credentials from the EC2 instance metadata service."""

    type: Literal["imds"] = "imds"


class AwsCredentialsDefault(_AwsCredentials):
    """Default credential provider chain."""

    type: Literal["default"] = "default"
    profile: str | None = Field(
        default=None, description='AWS profile name. Default: `"default"`.'
    )


class AwsCredentialsProfile(_AwsCredentials):
    """Credentials from a shared config/credentials file."""

    type: Literal["profile"] = "profile"
    profile: str | None = Field(
        default=None, description='AWS profile name. Default: `"default"`.'
    )
    config_file: str | None = Field(
        default=None, description='Config file. Default: `"${HOME}/.aws/config"`.'
    )
    credentials_file: str | None = Field(
        default=None,
        description='Credentials file. Default: `"${HOME}/.aws/credentials"`.',
    )


class AwsCredentialsEcs(_AwsCredentials):
    """Credentials from the ECS container credential provider."""

    type: Literal["ecs"] = "ecs"
    endpoint: str | None = Field(default=None, description="Credential endpoint.")
    auth_token_file: str | None = Field(
        default=None, description="File containing the authorization token."
    )


AwsCredentials: TypeAlias = Annotated[
    AwsCredentialsAnonymous
    | AwsCredentialsEnvironment
    | AwsCredentialsImds
    | AwsCredentialsDefault
    | AwsCredentialsProfile
    | AwsCredentialsEcs,
    Field(discriminator="type"),
]


class Context(TensorStoreModel):
    """TensorStore context specification.

    Maps resource identifiers to resource specs. A string value refers to another
    named resource (e.g. `"cache_pool#remote"`). Keys of the form
    `<resource-type>#<id>` define named resources and are accepted as extra fields.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    cache_pool: CachePool | str | None = None
    data_copy_concurrency: DataCopyConcurrency | str | None = None
    file_io_concurrency: FileIOConcurrency | str | None = None
    file_io_sync: bool | str | None = Field(
        default=None, description="Sync files after writing. Default: `true`."
    )
    file_io_mode: FileIOMode | str | None = Field(
        default=None, json_schema_extra=since("0.1.77")
    )
    file_io_locking: FileIOLocking | str | None = None
    http_request_concurrency: HTTPRequestConcurrency | str | None = None
    http_request_retries: RequestRetries | str | None = None
    gcs_request_concurrency: HTTPRequestConcurrency | str | None = None
    gcs_request_retries: RequestRetries | str | None = None
    gcs_user_project: GCSUserProject | str | None = None
    experimental_gcs_rate_limiter: RateLimiter | str | None = None
    s3_request_concurrency: HTTPRequestConcurrency | str | None = None
    s3_request_retries: RequestRetries | str | None = None
    experimental_s3_rate_limiter: RateLimiter | str | None = None
    aws_credentials: AwsCredentials | str | None = Field(
        default=None, json_schema_extra=since("0.1.72")
    )
    memory_key_value_store: MemoryKeyValueStore | str | None = None
    ocdbt_coordinator: OcdbtCoordinator | str | None = None
