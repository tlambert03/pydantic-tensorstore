"""Base TensorStore specification models.

Defines the main TensorStoreSpec class and driver registry system.
"""

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypeAlias

from annotated_types import Interval
from pydantic import BaseModel, ConfigDict, Field

from pydantic_tensorstore._types import ContextResource, DataType
from pydantic_tensorstore.core.context import Context
from pydantic_tensorstore.core.schema import Schema
from pydantic_tensorstore.core.transform import IndexTransform
from pydantic_tensorstore.kvstore import KvStore

if TYPE_CHECKING:
    import tensorstore


class BaseSpec(BaseModel):
    """Base class for all TensorStore Specs."""

    # REQUIRED IN ALL SUBCLASSES
    # omitted for the sake of type-hinting (so subclasses can use Literal types)
    # driver: str = Field(description="TensorStore driver identifier")

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        serialize_by_alias=True,
    )
    context: Context | None = Field(
        default=None,
        description="Context resource configuration",
    )
    dtype: DataType | None = Field(
        default=None,
        description="Specifies the data type.",
    )
    rank: Annotated[int, Interval(ge=0, le=32)] | None = Field(
        default=None,
        description=(
            "Specifies the rank of the TensorStore."
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

    def to_tensorstore(self) -> "tensorstore.Spec":
        """Instantiate a TensorStore object from the specification."""
        try:
            import tensorstore
        except ImportError as e:
            raise ImportError(
                "The tensorstore package is required to export to"
                " TensorStore specifications."
            ) from e

        data = self.model_dump(mode="json", exclude_unset=True)
        return tensorstore.Spec(data)


CacheRevalidationBound: TypeAlias = bool | Literal["open"] | float


class TensorStoreKvStoreAdapterSpec(BaseSpec):
    """Specifies a TensorStore stored using a base key-value store."""

    # driver: str
    kvstore: KvStore = Field(
        description="Key-value store for data storage",
    )

    path: str = Field(
        default="",
        description="Additional path relative to kvstore.",
    )
    cache_pool: ContextResource = "cache_pool"
    data_copy_concurrency: ContextResource = "data_copy_concurrency"
    recheck_cached_data: CacheRevalidationBound = Field(
        default="open",
        description=(
            "Time after which cached data is assumed to be fresh.  "
            "Cached data older than the specified time is revalidated prior to being "
            "returned from a read operation. Writes are always consistent regardless "
            "of the value of this option. "
            "Specifying true means that the data will be revalidated prior to every "
            "read operation. With a value of 'open', any cached data is revalidated "
            "when the TensorStore is opened but is not rechecked for each read "
            "operation."
        ),
    )


class ChunkedTensorStoreKvStoreAdapterSpec(TensorStoreKvStoreAdapterSpec):
    """Common options supported by all chunked storage drivers."""

    open: bool | None = None
    create: bool = False
    delete_existing: bool = False
    assume_metadata: bool = False
    assume_cached_metadata: bool = False
    metadata_cache_pool: ContextResource | None = None
    recheck_cached_metadata: CacheRevalidationBound = Field(
        default="open",
        description="Time after which cached metadata is assumed to be fresh.",
    )
    fill_missing_data_reads: bool = True
    store_data_equal_to_fill_value: bool = False
