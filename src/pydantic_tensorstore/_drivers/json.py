"""JSON driver: a rank-0 TensorStore backed by a JSON file."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._core.spec import TensorStoreKvStoreAdapterSpec
from pydantic_tensorstore._types import DataType


class JsonSpec(TensorStoreKvStoreAdapterSpec):
    """Rank-0 TensorStore of `json` data type stored in a JSON file."""

    driver: Literal["json"] = "json"
    dtype: Literal[DataType.JSON] | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    rank: Literal[0] | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    json_pointer: str | None = Field(
        default=None,
        description="JSON Pointer (RFC 6901) to a sub-value within the file. "
        'Default: `""` (the whole file).',
    )
