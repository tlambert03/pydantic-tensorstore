"""Schema model for TensorStore specifications.

The Schema defines the structure of data including data type, domain,
chunk layout, codec, fill value, and dimension units.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from pydantic_tensorstore.core.chunk_layout import ChunkLayout  # noqa: TC001
from pydantic_tensorstore.core.transform import IndexDomain  # noqa: TC001
from pydantic_tensorstore.types.common import DataType, DimensionIndex, Unit


class Schema(BaseModel):
    """TensorStore schema specification.

    Defines the structure and constraints for TensorStore data including
    data type, domain, chunking, encoding, and physical units.

    Attributes
    ----------
        dtype: Data type of array elements
        domain: Index domain (shape and labels)
        chunk_layout: Chunking configuration
        codec: Compression and encoding settings
        fill_value: Default value for unwritten elements
        dimension_units: Physical units for each dimension
        rank: Number of dimensions (computed from domain if not specified)

    Example:
        >>> schema = Schema(
        ...     dtype="float32",
        ...     domain={"shape": [100, 200]},
        ...     chunk_layout={"grid_origin": [0, 0]},
        ...     fill_value=0.0,
        ... )
    """

    model_config = {"extra": "forbid", "validate_assignment": True}

    dtype: DataType | None = Field(
        default=None, description="Data type of array elements"
    )

    domain: IndexDomain | None = Field(
        default=None, description="Index domain (shape, bounds, labels)"
    )

    chunk_layout: ChunkLayout | None = Field(
        default=None, description="Chunk layout constraints"
    )

    codec: Any | None = Field(
        default=None, description="Codec specification for compression/encoding"
    )

    fill_value: int | float | bool | str | list[Any] | None = Field(
        default=None, description="Fill value for unwritten elements"
    )

    dimension_units: list[str | Unit | None] | None = Field(
        default=None, description="Physical units for each dimension"
    )

    rank: DimensionIndex | None = Field(
        default=None, description="Number of dimensions"
    )

    @field_validator("dimension_units", mode="before")
    @classmethod
    def validate_dimension_units(cls, v: Any) -> list[Unit | None] | None:
        """Convert string units to Unit objects."""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("dimension_units must be a list")

        result: list[Unit | None] = []
        for unit in v:
            if unit is None:
                result.append(None)
            elif isinstance(unit, str):
                # Parse string units like "4nm", "1.5m", etc.
                if unit == "":
                    result.append(Unit(multiplier=1.0, base_unit=""))
                else:
                    # Simple parsing - in real implementation might use proper parser
                    import re

                    match = re.match(r"^([\d.]+)?([a-zA-Z]*)$", unit)
                    if match:
                        multiplier_str, base_unit = match.groups()
                        multiplier = float(multiplier_str) if multiplier_str else 1.0
                        result.append(Unit(multiplier=multiplier, base_unit=base_unit))
                    else:
                        result.append(Unit(multiplier=1.0, base_unit=unit))
            elif isinstance(unit, dict):
                result.append(Unit.model_validate(unit))
            else:
                result.append(unit)

        return result

    @field_validator("fill_value", mode="before")
    @classmethod
    def validate_fill_value(cls, v: Any) -> Any:
        """Validate fill value compatibility with data type."""
        # In a full implementation, this would check compatibility with dtype
        # For now, just ensure it's a reasonable JSON-serializable value
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        # Validate rank consistency
        if self.domain and self.rank is not None:
            domain_rank = len(self.domain.shape) if self.domain.shape else None
            if domain_rank is not None and domain_rank != self.rank:
                raise ValueError(
                    f"Specified rank {self.rank} doesn't match domain rank "
                    f"{domain_rank}"
                )

        # Set rank from domain if not specified
        if self.rank is None and self.domain and self.domain.shape:
            self.rank = len(self.domain.shape)

        # Validate dimension_units length
        if self.dimension_units is not None and self.rank is not None:
            if len(self.dimension_units) != self.rank:
                raise ValueError(
                    f"dimension_units length {len(self.dimension_units)} "
                    f"doesn't match rank {self.rank}"
                )

    def get_effective_rank(self) -> DimensionIndex | None:
        """Get the effective rank, computed from various sources."""
        if self.rank is not None:
            return self.rank
        if self.domain and self.domain.shape:
            return len(self.domain.shape)
        if self.dimension_units:
            return len(self.dimension_units)
        return None

    def is_compatible_with(self, other: Schema) -> bool:
        """Check if this schema is compatible with another schema."""
        # Check data type compatibility
        if self.dtype is not None and other.dtype is not None:
            if self.dtype != other.dtype:
                return False

        # Check rank compatibility
        self_rank = self.get_effective_rank()
        other_rank = other.get_effective_rank()
        if self_rank is not None and other_rank is not None:
            if self_rank != other_rank:
                return False

        # Domain compatibility would require more complex logic
        # For now, assume compatible if no conflicts
        return True
