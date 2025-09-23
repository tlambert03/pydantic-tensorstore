"""Index transform and domain models for TensorStore.

Index transforms map from input coordinates to output coordinates,
supporting operations like slicing, transposition, and broadcasting.
"""

from typing import Annotated, Any, Self

from annotated_types import Interval
from pydantic import BaseModel, Field, field_validator, model_validator

from pydantic_tensorstore._types import Shape


class DimensionSpec(BaseModel):
    """Specification for a single dimension."""

    model_config = {"extra": "forbid"}

    inclusive_min: int | None = Field(default=None, description="Inclusive lower bound")
    exclusive_max: int | None = Field(default=None, description="Exclusive upper bound")
    size: int | None = Field(default=None, description="Size of the dimension")
    label: str | None = Field(default=None, description="Dimension label")

    @field_validator("size", mode="before")
    @classmethod
    def validate_size(cls, v: Any) -> int | None:
        """Validate dimension size."""
        if v is None:
            return None
        if not isinstance(v, int):
            try:
                v_int = int(v)
            except (ValueError, TypeError):
                raise ValueError("Dimension size must be an integer") from None
        else:
            v_int = v
        if v_int <= 0:
            raise ValueError("Dimension size must be positive")
        return v_int

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        # Ensure consistency between bounds and size
        if (
            self.inclusive_min is not None
            and self.exclusive_max is not None
            and self.size is not None
        ):
            computed_size = self.exclusive_max - self.inclusive_min
            if computed_size != self.size:
                raise ValueError(
                    f"Size {self.size} inconsistent with bounds "
                    f"[{self.inclusive_min}, {self.exclusive_max})"
                )


class IndexDomain(BaseModel):
    """Index domain specification.

    Defines the coordinate space for array indexing, including
    bounds, labels, and implicit dimensions.

    Attributes
    ----------
        shape: Size of each dimension
        inclusive_min: Lower bounds (inclusive)
        exclusive_max: Upper bounds (exclusive)
        labels: Human-readable dimension labels
        implicit: Whether dimensions are implicit

    Example:
        >>> domain = IndexDomain(shape=[100, 200], labels=["height", "width"])
    """

    model_config = {"extra": "forbid", "validate_assignment": True}

    shape: Shape | None = Field(default=None, description="Shape of each dimension")

    inclusive_min: list[int] | None = Field(
        default=None, description="Inclusive lower bounds"
    )

    exclusive_max: list[int] | None = Field(
        default=None, description="Exclusive upper bounds"
    )

    labels: list[str | None] | None = Field(
        default=None, description="Dimension labels"
    )

    implicit: list[bool] | None = Field(
        default=None, description="Implicit dimension flags"
    )

    @field_validator("shape", mode="before")
    @classmethod
    def validate_shape(cls, v: Any) -> Shape | None:
        """Validate shape values."""
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("Shape must be a list")
        for i, dim_size in enumerate(v):
            if dim_size <= 0:
                raise ValueError(
                    f"Shape dimension {i} must be positive, got {dim_size}"
                )
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate consistency between different representations."""
        rank = None

        # Determine rank from various sources
        if self.shape is not None:
            rank = len(self.shape)

        if self.inclusive_min is not None:
            min_rank = len(self.inclusive_min)
            if rank is None:
                rank = min_rank
            elif rank != min_rank:
                raise ValueError(
                    f"inclusive_min length {min_rank} doesn't match rank {rank}"
                )

        if self.exclusive_max is not None:
            max_rank = len(self.exclusive_max)
            if rank is None:
                rank = max_rank
            elif rank != max_rank:
                raise ValueError(
                    f"exclusive_max length {max_rank} doesn't match rank {rank}"
                )

        if self.labels is not None:
            labels_rank = len(self.labels)
            if rank is None:
                rank = labels_rank
            elif rank != labels_rank:
                raise ValueError(
                    f"labels length {labels_rank} doesn't match rank {rank}"
                )

        if self.implicit is not None:
            implicit_rank = len(self.implicit)
            if rank is None:
                rank = implicit_rank
            elif rank != implicit_rank:
                raise ValueError(
                    f"implicit length {implicit_rank} doesn't match rank {rank}"
                )

        # Validate bounds consistency with shape
        if (
            self.shape is not None
            and self.inclusive_min is not None
            and self.exclusive_max is not None
        ):
            for i, (size, min_val, max_val) in enumerate(
                zip(self.shape, self.inclusive_min, self.exclusive_max, strict=False)
            ):
                computed_size = max_val - min_val
                if computed_size != size:
                    raise ValueError(
                        f"Dimension {i}: size {size} inconsistent with bounds "
                        f"[{min_val}, {max_val})"
                    )

    @property
    def rank(self) -> int | None:
        """Get the rank (number of dimensions)."""
        if self.shape is not None:
            return len(self.shape)
        if self.inclusive_min is not None:
            return len(self.inclusive_min)
        if self.exclusive_max is not None:
            return len(self.exclusive_max)
        if self.labels is not None:
            return len(self.labels)
        if self.implicit is not None:
            return len(self.implicit)
        return None


class OutputIndexMap(BaseModel):
    """Output index map for index transforms."""

    model_config = {"extra": "forbid"}

    input_dimension: int | None = Field(
        default=None, description="Input dimension index"
    )
    offset: int = Field(default=0, description="Offset value")
    stride: int = Field(default=1, description="Stride value")
    index_array: list[int] | None = Field(
        default=None, description="Index array for advanced indexing"
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate output index map configuration."""
        if self.input_dimension is not None and self.index_array is not None:
            raise ValueError("Cannot specify both input_dimension and index_array")
        if self.input_dimension is None and self.index_array is None:
            raise ValueError("Must specify either input_dimension or index_array")
        if self.stride == 0:
            raise ValueError("Stride cannot be zero")


class IndexTransform(BaseModel):
    """Index transform specification.

    Maps coordinates from an input space to an output space,
    supporting operations like slicing, broadcasting, and reordering.
    """

    model_config = {"extra": "forbid", "validate_assignment": True}

    input_rank: Annotated[int, Interval(ge=0, le=32)] | None = Field(
        default=None, description="Number of input dimensions."
    )
    input_inclusive_min: list[int | list[int]] | None = Field(
        default=None, description="Inclusive lower bounds of the input domain."
    )

    input_exclusive_max: list[int | list[int]] | None = Field(
        default=None, description="Exclusive upper bounds of the input domain."
    )

    input_inclusive_max: list[int | list[int]] | None = Field(
        default=None, description="Inclusive upper bounds of the input domain."
    )

    output_rank: int | None = Field(default=None, description="Output rank")

    input_shape: Shape | None = Field(default=None, description="Input domain shape")

    input_labels: list[str | None] | None = Field(
        default=None, description="Input dimension labels"
    )

    output: list[OutputIndexMap] | None = Field(
        default=None, description="Output index maps"
    )

    @model_validator(mode="after")
    def _post_validate(self) -> Self:
        """Validate transform consistency."""
        # Validate that at most one of input_exclusive_max, input_inclusive_max,
        # and input_shape is specified
        bound_specs = [
            self.input_exclusive_max is not None,
            self.input_inclusive_max is not None,
            self.input_shape is not None,
        ]
        if sum(bound_specs) > 1:
            raise ValueError(
                "At most one of input_exclusive_max, input_inclusive_max, "
                "and input_shape may be specified"
            )

        # Determine input rank
        input_rank = self.input_rank
        if input_rank is None and self.input_shape is not None:
            input_rank = len(self.input_shape)
        if input_rank is None and self.input_labels is not None:
            input_rank = len(self.input_labels)
        if input_rank is None and self.input_inclusive_min is not None:
            input_rank = len(self.input_inclusive_min)
        if input_rank is None and self.input_exclusive_max is not None:
            input_rank = len(self.input_exclusive_max)
        if input_rank is None and self.input_inclusive_max is not None:
            input_rank = len(self.input_inclusive_max)

        # Determine output rank
        output_rank = self.output_rank
        if output_rank is None and self.output is not None:
            output_rank = len(self.output)

        # Validate consistency
        if input_rank is not None:
            if self.input_rank is not None and self.input_rank != input_rank:
                raise ValueError("Inconsistent input rank specification")
            if self.input_shape is not None and len(self.input_shape) != input_rank:
                raise ValueError("input_shape length doesn't match input rank")
            if self.input_labels is not None and len(self.input_labels) != input_rank:
                raise ValueError("input_labels length doesn't match input rank")
            if (
                self.input_inclusive_min is not None
                and len(self.input_inclusive_min) != input_rank
            ):
                raise ValueError("input_inclusive_min length doesn't match input rank")
            if (
                self.input_exclusive_max is not None
                and len(self.input_exclusive_max) != input_rank
            ):
                raise ValueError("input_exclusive_max length doesn't match input rank")
            if (
                self.input_inclusive_max is not None
                and len(self.input_inclusive_max) != input_rank
            ):
                raise ValueError("input_inclusive_max length doesn't match input rank")

        if output_rank is not None:
            if self.output_rank is not None and self.output_rank != output_rank:
                raise ValueError("Inconsistent output rank specification")
            if self.output is not None and len(self.output) != output_rank:
                raise ValueError("output length doesn't match output rank")

        # Validate output index maps reference valid input dimensions
        if self.output is not None and input_rank is not None:
            for i, output_map in enumerate(self.output):
                if (
                    output_map.input_dimension is not None
                    and output_map.input_dimension >= input_rank
                ):
                    raise ValueError(
                        f"Output {i} references input dimension "
                        f"{output_map.input_dimension} >= input rank {input_rank}"
                    )
        return self
