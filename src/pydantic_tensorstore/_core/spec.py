"""Base TensorStore specification models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from annotated_types import Interval
from pydantic import Field

from pydantic_tensorstore._core.base import TensorStoreModel, since
from pydantic_tensorstore._core.context import Context
from pydantic_tensorstore._core.schema import Schema
from pydantic_tensorstore._core.transform import IndexTransform
from pydantic_tensorstore._kvstore import KvStore
from pydantic_tensorstore._types import DTYPE_SINCE, ContextResource, DataType

if TYPE_CHECKING:
    import tensorstore

    from pydantic_tensorstore._drivers import TensorStoreSpec  # noqa: TC004


class BaseSpec(TensorStoreModel):
    """Base class for all TensorStore Specs."""

    # REQUIRED IN ALL SUBCLASSES
    # omitted for the sake of type-hinting (so subclasses can use Literal types)
    # driver: str = Field(description="TensorStore driver identifier")

    context: Context | None = Field(
        default=None,
        description="Context resource configuration",
    )
    dtype: DataType | None = Field(
        default=None,
        description="Specifies the data type.",
        json_schema_extra=since(DTYPE_SINCE),
    )
    rank: Annotated[int, Interval(ge=0, le=32)] | None = Field(
        default=None,
        description=(
            "Specifies the rank of the TensorStore. "
            "If transform is also specified, the input rank must match. Otherwise, the "
            "rank constraint applies to the driver directly."
        ),
    )
    transform: IndexTransform | None = Field(
        default=None,
        description="Specifies an index transform to apply.",
    )
    schema_: Schema | None = Field(
        default=None,
        description="Schema constraints",
        alias="schema",
    )

    def to_tensorstore(self, *, check_version: bool = True) -> tensorstore.Spec:
        """Instantiate a `tensorstore.Spec` from the specification.

        Parameters
        ----------
        check_version : bool
            If True (default), first verify that the installed tensorstore supports
            every feature used, raising `UnsupportedTensorStoreVersionError` with
            the offending fields otherwise (instead of tensorstore's own, less
            specific, "extra members" error).
        """
        try:
            import tensorstore
        except ImportError as e:
            raise ImportError(
                "The tensorstore package is required to export to"
                " TensorStore specifications."
            ) from e

        if check_version:
            self.check_tensorstore_version()
        return tensorstore.Spec(self.model_dump(mode="json"))


CacheRevalidationBound: TypeAlias = bool | Literal["open"] | float
"""Bound on cache staleness.

`true` revalidates before every read, `false` never revalidates, `"open"` revalidates
once when the TensorStore is opened, and a number is a Unix timestamp in seconds.
"""


class TensorStoreKvStoreAdapterSpec(BaseSpec):
    """Specifies a TensorStore stored using a base key-value store."""

    # driver: str
    kvstore: KvStore = Field(description="Key-value store for data storage")
    path: str | None = Field(
        default=None,
        description="Additional path relative to kvstore. Deprecated: specify the "
        "path in `kvstore.path` instead.",
        deprecated=True,
    )
    cache_pool: ContextResource | None = Field(
        default=None,
        description='Cache pool for data. Default: `"cache_pool"`.',
    )
    data_copy_concurrency: ContextResource | None = Field(
        default=None,
        description="Concurrency limit for data copying. "
        'Default: `"data_copy_concurrency"`.',
    )
    recheck_cached_data: CacheRevalidationBound | None = Field(
        default=None,
        description=(
            "Time after which cached data is assumed to be fresh. Cached data older "
            "than the specified time is revalidated prior to being returned from a "
            "read operation. Writes are always consistent regardless of the value of "
            'this option. Default: `"open"`.'
        ),
    )


class ChunkedTensorStoreKvStoreAdapterSpec(TensorStoreKvStoreAdapterSpec):
    """Common options supported by all chunked storage drivers."""

    open: bool | None = Field(
        default=None,
        description="Open an existing TensorStore. If neither `open` nor `create` is "
        "specified, defaults to `true`.",
    )
    create: bool | None = Field(
        default=None,
        description="Create a new TensorStore if one does not exist. Default: `false`.",
    )
    delete_existing: bool | None = Field(
        default=None,
        description="Delete any existing data before creating a new array. "
        "Requires `create`. Default: `false`.",
    )
    assume_metadata: bool | None = Field(
        default=None,
        description="Skip reading the metadata when opening; the metadata specified "
        "in the spec is assumed to be correct. Default: `false`.",
    )
    assume_cached_metadata: bool | None = Field(
        default=None,
        description="Skip reading the metadata when opening if it is already cached. "
        "Default: `false`.",
    )
    metadata_cache_pool: ContextResource | None = Field(
        default=None,
        description="Cache pool for metadata only. Defaults to `cache_pool`.",
    )
    recheck_cached_metadata: CacheRevalidationBound | None = Field(
        default=None,
        description="Time after which cached metadata is assumed to be fresh. "
        'Default: `"open"`.',
    )
    recheck_cached_data: CacheRevalidationBound | None = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default=None,
        description="Time after which cached data is assumed to be fresh. "
        "Default: `true`.",
    )
    fill_missing_data_reads: bool | None = Field(
        default=None,
        description="Replace missing chunks with the fill value when reading. "
        "Default: `true`. (tensorstore >= 0.1.68)",
    )
    store_data_equal_to_fill_value: bool | None = Field(
        default=None,
        description="Store chunks even if they are equal to the fill value. "
        "Default: `false`. (tensorstore >= 0.1.68)",
    )


class TensorStoreAdapterSpec(BaseSpec):
    """Base for drivers that adapt another TensorStore (or a kvstore)."""

    base: TensorStoreSpec | KvStore = Field(
        description="Underlying TensorStore (or kvstore, for drivers that accept one)."
    )
