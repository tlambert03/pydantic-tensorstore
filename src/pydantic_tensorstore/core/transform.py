"""Index transform and domain models for TensorStore.

Index transforms map from input coordinates to output coordinates,
supporting operations like slicing, transposition, and broadcasting.
"""

from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Literal

from annotated_types import Interval
from pydantic import BaseModel, Field, NonNegativeInt, field_validator, model_validator
from typing_extensions import Self

from pydantic_tensorstore._types import Shape


def _validate_labels(cls: type, v: Any) -> Any:
    # Non-empty strings must not occur more than once
    if isinstance(v, Sequence):
        seen = set()
        for label in v:
            if label:
                if label in seen:
                    raise ValueError(f"Duplicate label: {label}")
                seen.add(label)
        return list(v)
    return v


class IndexDomain(BaseModel):
    """Index domain specification.

    Defines the coordinate space for array indexing, including
    bounds, labels, and implicit dimensions.
    """

    model_config: ClassVar = {"extra": "forbid", "validate_assignment": True}

    rank: Annotated[int, Interval(ge=0, le=32)] | None = Field(
        default=None, description="Number of dimensions"
    )

    inclusive_min: list[int | list[int]] | None = Field(
        default=None, description="Inclusive lower bounds of the domain."
    )

    exclusive_max: list[int | list[int]] | None = Field(
        default=None, description="Exclusive upper bounds of the domain."
    )

    inclusive_max: list[int | list[int]] | None = Field(
        default=None, description="Inclusive upper bounds of the domain."
    )

    shape: Shape | None = Field(default=None, description="Shape of each dimension")

    labels: list[str] | None = Field(default=None, description="Dimension labels")

    _v: Any = field_validator("labels", mode="before")(classmethod(_validate_labels))

    @property
    def effective_rank(self) -> int:
        """Effective rank.

        The rank must be specified either directly, or implicitly by the number of
        dimensions specified for inclusive_min, inclusive_max, exclusive_max, shape, or
        labels.
        """
        if self.rank is not None:
            return self.rank
        for attr in ("inclusive_min", "exclusive_max", "shape", "labels"):
            if (val := getattr(self, attr)) is not None:
                return len(val)
        raise ValueError(
            'At least one of "rank", "inclusive_min", "exclusive_max", "shape", or '
            '"labels" must be specified'
        )

    @model_validator(mode="after")
    def _post_validate(self) -> Self:
        """Validate domain consistency."""
        rank = self.effective_rank  # raises if invalid

        # Validate consistency
        bad_fields: list[str] = []
        for field in [
            "shape",
            "labels",
            "inclusive_min",
            "exclusive_max",
            "inclusive_max",
        ]:
            if (val := getattr(self, field)) is not None and len(val) != rank:
                bad_fields.append(field)

        if bad_fields:
            msg = ", ".join(
                f"'{field}' length {len(getattr(self, field))}" for field in bad_fields
            )
            raise ValueError(f"{msg} don't match rank {rank}")

        return self


IndexInterval = tuple[int | Literal["-inf"], int | Literal["+inf"]]
"""Specifies a closed interval of integer index values."""


class OutputIndexMap(BaseModel):
    """Output index map for index transforms."""

    model_config: ClassVar = {"extra": "forbid"}

    offset: int | None = Field(default=None, description="Offset value")
    stride: int | None = Field(default=None, description="Stride value")

    input_dimension: NonNegativeInt | None = Field(
        default=None, description="Input dimension index"
    )
    index_array: list[int] | int | None = Field(
        default=None, description="Index array for advanced indexing"
    )
    index_array_bounds: IndexInterval | None = Field(
        default=None,
        description="""If present, specifies constraints on the values within
index_array (which must also be specified). If index_array contains an out-of-bounds
index, an error may not be returned immediately but will be returned if the
corresponding position within the domain is accessed. If the indices in index_array have
already been validated, this need not be specified. This allows transforms containing
out-of-bounds index array indices to correctly round trip through JSON, but normally
need not be specified manually.
""",
    )

    @model_validator(mode="after")
    def _post_validator(self) -> Self:
        """Validate output index map configuration."""
        # Check that input_dimension and index_array are mutually exclusive
        if self.input_dimension is not None and self.index_array is not None:
            raise ValueError("input_dimension and index_array cannot both be specified")

        # Check that stride is only specified with input_dimension or index_array
        if self.stride is not None and (
            self.input_dimension is None and self.index_array is None
        ):
            raise ValueError(
                "stride can only be specified with input_dimension or index_array"
            )

        # Check that index_array_bounds is only specified with index_array
        if self.index_array_bounds is not None and self.index_array is None:
            raise ValueError(
                "index_array_bounds can only be specified with index_array"
            )

        return self


class IndexTransform(BaseModel):
    """Index transform specification.

    Maps coordinates from an input space to an output space,
    supporting operations like slicing, broadcasting, and reordering.
    """

    model_config: ClassVar = {"extra": "forbid", "validate_assignment": True}

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

    input_shape: Shape | None = Field(default=None, description="Input domain shape")

    input_labels: list[str] | None = Field(
        default=None, description="Input dimension labels"
    )

    output: list[OutputIndexMap] | None = Field(
        default=None, description="Output index maps"
    )

    _v: Any = field_validator("input_labels", mode="before")(
        classmethod(_validate_labels)
    )

    @property
    def effective_rank(self) -> int:
        """Input rank, inferred if not explicitly set."""
        if self.input_rank is not None:
            return self.input_rank

        for attr in (
            "input_inclusive_min",
            "input_exclusive_max",
            "input_shape",
            "input_labels",
        ):
            if (val := getattr(self, attr)) is not None:
                return len(val)
        raise ValueError(
            'At least one of "input_rank", "input_inclusive_min", "input_shape", '
            '"input_inclusive_max", "input_exclusive_max", "input_labels" members '
            "must be specified"
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
                "At most one of 'input_exclusive_max', 'input_inclusive_max', "
                "and 'input_shape' may be specified"
            )

        input_rank = self.effective_rank  # raises if invalid

        # Validate consistency
        bad_fields: list[str] = []
        for field in [
            "input_shape",
            "input_labels",
            "input_inclusive_min",
            "input_exclusive_max",
            "input_inclusive_max",
        ]:
            if (val := getattr(self, field)) is not None and len(val) != input_rank:
                bad_fields.append(field)

        if bad_fields:
            msg = ", ".join(
                f"'{field}' length {len(getattr(self, field))}" for field in bad_fields
            )
            raise ValueError(f"{msg} don't match input rank {input_rank}")

        # Determine output rank from output list
        if self.output is not None:
            # Validate output index maps reference valid input dimensions
            if input_rank is not None:
                for i, output_map in enumerate(self.output):
                    input_dim = output_map.input_dimension
                    if input_dim is not None and input_dim >= input_rank:
                        raise ValueError(
                            f"Output {i} references input dimension "
                            f"{input_dim} >= input rank {input_rank}"
                        )
                    # If the input_rank is 0, output.index_array must be a numeric value
                    if input_rank == 0:
                        idx_arr = output_map.index_array
                        if idx_arr is not None and not isinstance(idx_arr, int):
                            raise ValueError(
                                "output.index_array must be an integer when "
                                "input_rank is 0."
                            )
        return self
