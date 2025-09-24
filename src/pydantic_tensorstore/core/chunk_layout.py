"""Chunk layout specification for TensorStore.

Defines how data is partitioned into chunks for storage and I/O optimization.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from annotated_types import Ge, Interval
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, model_validator


class ChunkLayoutGrid(BaseModel):
    """Constraints on the write/read/codec chunk grids."""

    shape: list[NonNegativeInt] | Literal[-1] | None = Field(
        default=None,
        description=(
            """Hard constraints on the chunk size for each dimension.

The length must equal the rank of the index space. Each element constrains the chunk
size for the corresponding dimension, and must be a non-negative integer. The special
value of 0 (or, equivalently, null)for a given dimension indicates no constraint. The
special value of -1 for a given dimension indicates that the chunk size should equal the
full extent of the domain, and is always treated as a soft constraint."""
        ),
    )
    shape_soft_constraint: list[NonNegativeInt] | Literal[-1] | None = Field(
        default=None,
        description="Preferred chunk sizes for each dimension. If a non-zero, "
        "non-null size for a given dimension is specified in both shape and "
        "shape_soft_constraint, shape takes precedence.",
    )
    aspect_ratio: list[NonNegativeFloat] | None = Field(
        default=None,
        description=(
            """Aspect ratio of the chunk shape.

Specifies the relative chunk size along each dimension. The special value of 0 (or,
equivalently, null) indicates no preference (which results in the default aspect ratio
of 1 if not otherwise specified). The aspect ratio preference is only taken into account
if the chunk size along a given dimension is not specified by shape or
shape_soft_constraint, or otherwise constrained. For example, an aspect_ratio of [1,
1.5, 1.5] indicates that the chunk size along dimensions 1 and 2 should be 1.5 times the
chunk size along dimension 0. If the target number of elements is 486000, then the
resultant chunk size will be [60, 90, 90] (assuming it is not otherwise constrained).
"""
        ),
    )
    aspect_ratio_soft_constraint: list[NonNegativeFloat] | None = Field(
        default=None,
        description=(
            "Soft constraint on aspect ratio, lower precedence than aspect_ratio."
        ),
    )
    elements: Annotated[int, Ge(1)] | None = Field(
        default=None,
        description=(
            "Preferred number of elements per chunk. "
            "Used in conjunction with aspect_ratio to determine the chunk size for "
            "dimensions that are not otherwise constrained. The special value of null "
            "indicates no preference, in which case a driver-specific default may "
            "be used."
        ),
    )
    elements_soft_constraint: Annotated[int, Ge(1)] | None = Field(
        default=None,
        description=(
            "Preferred number of elements per chunk, lower precedence than elements."
        ),
    )

    @model_validator(mode="after")
    def _validate_array_lengths_consistent(self) -> ChunkLayoutGrid:
        """Validate that all array fields have consistent lengths."""
        arrays: list[int] = []
        array_names: list[str] = []
        field_names = [
            "shape",
            "shape_soft_constraint",
            "aspect_ratio",
            "aspect_ratio_soft_constraint",
        ]
        for field_name in field_names:
            value = getattr(self, field_name)
            if isinstance(value, list):
                arrays.append(len(value))
                array_names.append(field_name)

        if len(set(arrays)) > 1:
            array_info = ", ".join(
                f"{name}={length}"
                for name, length in zip(array_names, arrays, strict=True)
            )
            raise ValueError(
                f"All array fields must have the same length (rank): {array_info}"
            )

        return self


class ChunkLayout(BaseModel):
    """Chunk layout specification.

    Controls how array data is partitioned into chunks for storage,
    compression, and parallel I/O.
    """

    model_config: ClassVar = {"extra": "forbid", "validate_assignment": True}

    rank: Annotated[int, Interval(ge=0, le=32)] | None = Field(
        default=None, description="Number of dimensions"
    )

    grid_origin: list[int | None] | None = Field(
        default=None,
        description="Specifies hard constraints on the origin of the chunk grid.",
    )

    grid_origin_soft_constraint: list[int | None] | None = Field(
        default=None,
        description="Specifies preferred values for the origin of the chunk grid "
        "rather than hard constraints.",
    )

    inner_order: list[int] | None = Field(
        default=None,
        description="Permutation specifying the element storage order within the "
        "innermost chunks.",
    )

    inner_order_soft_constraint: list[int] | None = Field(
        default=None,
        description="Specifies a preferred element storage order within the innermost "
        "chunks rather than a hard constraint.  `inner_order` takes precedence.",
    )

    write_chunk: ChunkLayoutGrid | None = Field(
        default=None,
        description="Constraints on the chunk grid over which writes may be "
        "efficiently partitioned.",
    )
    read_chunk: ChunkLayoutGrid | None = Field(
        default=None,
        description="Constraints on the chunk grid over which reads may be "
        "efficiently partitioned.",
    )
    codec_chunk: ChunkLayoutGrid | None = Field(
        default=None,
        description="Constraints on the chunk grid used by the codec, if applicable.",
    )
    chunk: ChunkLayoutGrid | None = Field(
        default=None,
        description=(
            """Combined constraints on write/read/codec chunks.

If aspect_ratio is specified, it applies to write_chunk, read_chunk, and codec_chunk. If
aspect_ratio_soft_constraint is specified, it also applies to write_chunk, read_chunk,
and codec_chunk, but with lower precedence than any write/read/codec-specific value that
is also specified.

If shape or elements is specified, it applies to write_chunk and read_chunk (but not
codec_chunk). If shape_soft_constraint or elements_soft_constraint is specified, it also
applies to write_chunk and read_chunk, but with lower precedence than any
write/read-specific value that is also specified."""
        ),
    )

    @model_validator(mode="after")
    def _post_validate(self) -> ChunkLayout:
        """Validate that inner_order is a valid permutation."""
        # validate_inner_order and inner_order_soft_constraint
        for field in ["inner_order", "inner_order_soft_constraint"]:
            if (v := getattr(self, field)) is not None:
                if self.rank is None:
                    raise ValueError(f"rank must be specified when {field} is provided")
                if sorted(v) != list(range(self.rank)):
                    raise ValueError(
                        f"{field} must be a permutation of "
                        f"[0, 1, ..., {self.rank - 1}], got {v}"
                    )

        # validate_grid_origin_length and grid_origin_soft_constraint_length
        for field in ["grid_origin", "grid_origin_soft_constraint"]:
            value = getattr(self, field)
            if value is not None and self.rank is not None and len(value) != self.rank:
                raise ValueError(
                    f"{field} length ({len(value)}) must equal rank ({self.rank})"
                )

        return self
