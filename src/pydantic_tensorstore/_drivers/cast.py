"""Cast driver: converts the data type of a base TensorStore."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._core.base import since
from pydantic_tensorstore._core.spec import TensorStoreAdapterSpec
from pydantic_tensorstore._types import DTYPE_SINCE, DataType


class CastSpec(TensorStoreAdapterSpec):
    """Virtual view that converts the data type of the base TensorStore."""

    driver: Literal["cast"] = "cast"
    dtype: DataType = Field(  # pyright: ignore
        description="Data type of the view.", json_schema_extra=since(DTYPE_SINCE)
    )
