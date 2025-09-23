"""Base TensorStore specification models.

Defines the main TensorStoreSpec class and driver registry system.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from annotated_types import Interval
from pydantic import BaseModel, ConfigDict, Field

from pydantic_tensorstore._types import DataType  # noqa: TC001
from pydantic_tensorstore.core.context import Context  # noqa: TC001
from pydantic_tensorstore.core.schema import Schema  # noqa: TC001
from pydantic_tensorstore.core.transform import IndexTransform  # noqa: TC001
from pydantic_tensorstore.kvstore import KvStore  # noqa: TC001


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


class TensorStoreKvStoreAdapterSpec(BaseSpec):
    """Specifies a TensorStore stored using a base key-value store."""

    # driver: str
    kvstore: KvStore | None = Field(
        default=None,
        description="Key-value store for data storage",
    )


class ChunkedTensorStoreKvStoreAdapterSpec(TensorStoreKvStoreAdapterSpec):
    """Common options supported by all chunked storage drivers."""
