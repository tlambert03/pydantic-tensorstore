"""Array driver specification for in-memory arrays."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, GetCoreSchemaHandler, model_validator
from pydantic_core import core_schema

from pydantic_tensorstore._core.spec import BaseSpec
from pydantic_tensorstore._types import ContextResource, DataType


class ArrayValidator:
    """Pydantic-compatible numpy array (validated from nested lists)."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _serialize(val: np.ndarray) -> list:
            return val.tolist()  # type: ignore[no-any-return]

        def _validate_array(val: Any) -> np.ndarray:
            return val if isinstance(val, np.ndarray) else np.asarray(val)

        ser_schema = core_schema.plain_serializer_function_ser_schema(
            _serialize, return_schema=core_schema.list_schema()
        )
        return core_schema.no_info_before_validator_function(
            _validate_array,
            core_schema.any_schema(),
            serialization=ser_schema,
        )


def _round_trips(original: np.ndarray, cast: np.ndarray) -> bool:
    """Whether casting back reproduces the original (treating NaN as equal)."""
    try:
        back = cast.astype(original.dtype)
    except (TypeError, ValueError):  # pragma: no cover - non-numeric round trip
        return True
    same = back == original
    if np.issubdtype(original.dtype, np.floating):
        same = same | (np.isnan(back) & np.isnan(original))
    return bool(np.all(same))


class ArraySpec(BaseSpec):
    """Array driver specification for in-memory arrays."""

    driver: Literal["array"] = "array"
    dtype: DataType | None = Field(
        default=None,
        description="Data type. Required, unless given instead via `schema.dtype`, "
        "or when nested as the `base` of a `downsample` driver (which hoists it, "
        "since downsampling preserves dtype; other adapters such as `cast` do "
        "not hoist, since they convert between dtypes).",
    )
    array: Annotated[np.ndarray, ArrayValidator] = Field(
        description="Nested array data or NumPy array",
    )
    data_copy_concurrency: ContextResource | None = Field(
        default=None,
        description='Concurrency for copying. Default: `"data_copy_concurrency"`.',
    )

    @model_validator(mode="after")
    def _validate_array(self) -> Self:
        """Cast to the declared dtype when numpy knows it; check rank."""
        effective_dtype = self.dtype
        if effective_dtype is None and self.schema_ is not None:
            effective_dtype = self.schema_.dtype
        try:
            np_dtype = (
                np.dtype(str(effective_dtype)) if effective_dtype is not None else None
            )
        except TypeError:
            np_dtype = None
        if np_dtype is not None and self.array.dtype != np_dtype:
            cast = self.array.astype(np_dtype)
            if not _round_trips(self.array, cast):
                raise ValueError(
                    f"array values cannot be represented in dtype '{effective_dtype}' "
                    f"without loss; cast the array explicitly if that is intended"
                )
            object.__setattr__(self, "array", cast)
        if self.rank is not None and self.rank != self.array.ndim:
            raise ValueError(
                f"Specified rank ({self.rank}) does not match array dimensions "
                f"({self.array.ndim})"
            )
        return self
