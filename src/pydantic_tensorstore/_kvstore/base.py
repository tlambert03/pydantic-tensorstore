"""Base key-value store specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypeAlias

from pydantic import Field, StringConstraints

from pydantic_tensorstore._core.base import TensorStoreModel
from pydantic_tensorstore._core.context import Context
from pydantic_tensorstore._types import ContextResource

if TYPE_CHECKING:
    from pydantic_tensorstore._kvstore import KvStore  # noqa: TC004

KvStoreUrl: TypeAlias = Annotated[
    str, StringConstraints(pattern=r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")
]
"""A kvstore URL such as `gs://bucket/path` or a pipeline like `memory://a.zip|zip:`."""


class BaseKvStore(TensorStoreModel):
    """Base class for key-value store specifications.

    Key-value stores provide the underlying storage layer for many TensorStore
    drivers, abstracting over local files, cloud storage, databases, etc.
    """

    # driver: str

    path: str | None = Field(
        default=None,
        description=(
            "Key prefix within the key-value store. If the prefix is intended "
            "to correspond to a Unix-style directory path, it should end with '/'."
        ),
    )
    context: Context | None = Field(
        default=None,
        description="Context resources that augment/override the parent context.",
    )


class KvStoreAdapter(BaseKvStore):
    """Base for kvstore drivers that wrap another kvstore."""

    base: KvStore = Field(description="Underlying key-value store.")


class _CachedKvStoreAdapter(KvStoreAdapter):
    """Adapter with the common `cache_pool`/`data_copy_concurrency` options."""

    cache_pool: ContextResource | None = Field(
        default=None, description='Cache pool for data. Default: `"cache_pool"`.'
    )
    data_copy_concurrency: ContextResource | None = Field(
        default=None,
        description='Concurrency for copying. Default: `"data_copy_concurrency"`.',
    )
