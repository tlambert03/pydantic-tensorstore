"""Key-value store specifications for TensorStore."""

from __future__ import annotations

from typing import Annotated, Any, TypeAlias
from urllib.parse import unquote, urlsplit

from pydantic import BeforeValidator, Field

from .base import BaseKvStore, KvStoreAdapter, KvStoreUrl
from .file import FileKvStore
from .gcs import GCSKvStore
from .http import HTTPKvStore
from .kvstack import KvStackKvStore, KvStackLayer
from .memory import MemoryKvStore
from .neuroglancer_uint64_sharded import (
    NeuroglancerShardingSpec,
    NeuroglancerUint64ShardedKvStore,
)
from .ocdbt import OcdbtConfig, OcdbtKvStore, OcdbtZstdCompression
from .s3 import S3KvStore
from .tsgrpc import TsGrpcKvStore
from .zarr3_sharding_indexed import Zarr3ShardingIndexedKvStore
from .zip import ZipKvStore

__all__ = [
    "BaseKvStore",
    "FileKvStore",
    "GCSKvStore",
    "HTTPKvStore",
    "KvStackKvStore",
    "KvStackLayer",
    "KvStore",
    "KvStoreAdapter",
    "KvStoreUrl",
    "MemoryKvStore",
    "NeuroglancerShardingSpec",
    "NeuroglancerUint64ShardedKvStore",
    "OcdbtConfig",
    "OcdbtKvStore",
    "OcdbtZstdCompression",
    "S3KvStore",
    "TsGrpcKvStore",
    "Zarr3ShardingIndexedKvStore",
    "ZipKvStore",
]


def _parse_kvstore_url(value: Any) -> Any:
    """Convert simple, single-driver URLs to dicts; pass anything else through.

    Paths are percent-decoded, matching tensorstore. tensorstore accepts URL
    strings natively (including pipelines such as `memory://a.zip|zip:`), so
    unknown forms are left as strings.
    """
    if not isinstance(value, str) or "|" in value:
        return value
    if value.startswith("file://"):
        return {"driver": "file", "path": unquote(value[len("file://") :])}
    if value.startswith("memory://"):
        path = unquote(value[len("memory://") :])
        return {"driver": "memory", **({"path": path} if path else {})}
    if value.startswith(("s3://", "gs://")):
        driver = "s3" if value.startswith("s3://") else "gcs"
        bucket, _, path = value.split("://", 1)[1].partition("/")
        path = unquote(path)
        return {"driver": driver, "bucket": bucket, **({"path": path} if path else {})}
    if value.startswith(("http://", "https://")):
        parts = urlsplit(value)
        base_url = parts._replace(path="").geturl()
        return {
            "driver": "http",
            "base_url": base_url,
            **({"path": parts.path} if parts.path else {}),
        }
    return value


_KvStoreModel: TypeAlias = Annotated[
    FileKvStore
    | GCSKvStore
    | HTTPKvStore
    | KvStackKvStore
    | MemoryKvStore
    | NeuroglancerUint64ShardedKvStore
    | OcdbtKvStore
    | S3KvStore
    | TsGrpcKvStore
    | Zarr3ShardingIndexedKvStore
    | ZipKvStore,
    Field(discriminator="driver"),
]

KvStore: TypeAlias = Annotated[
    _KvStoreModel | KvStoreUrl, BeforeValidator(_parse_kvstore_url)
]
"""Any kvstore spec: a driver model, or a URL string passed through to tensorstore."""

# Resolve the forward reference to `KvStore` in adapter models.
for _model in (
    KvStoreAdapter,
    KvStackLayer,
    NeuroglancerUint64ShardedKvStore,
    OcdbtKvStore,
    Zarr3ShardingIndexedKvStore,
    ZipKvStore,
):
    _model.model_rebuild()
