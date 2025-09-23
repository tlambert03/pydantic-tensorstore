"""Chunk layout specification for TensorStore.

Defines how data is partitioned into chunks for storage and I/O optimization.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from pydantic_tensorstore.types.common import ChunkShape, Shape  # noqa: TC001


class ChunkLayout(BaseModel):
    """Chunk layout specification.

    Controls how array data is partitioned into chunks for storage,
    compression, and parallel I/O.

    Attributes
    ----------
        grid_origin: Origin point for chunk grid alignment
        inner_order: Dimension order for memory layout within chunks
        read_chunk_shape: Shape for read operations
        write_chunk_shape: Shape for write operations
        chunk_shape: Shape for both read and write (convenience)
        chunk_elements: Target number of elements per chunk
        chunk_bytes: Target size in bytes per chunk
        chunk_aspect_ratio: Preferred aspect ratio for chunks

    Example:
        >>> layout = ChunkLayout(
        ...     chunk_shape=[64, 64, 64], inner_order=[0, 1, 2], grid_origin=[0, 0, 0]
        ... )
    """

    model_config = {"extra": "forbid", "validate_assignment": True}

    grid_origin: Shape | None = Field(
        default=None,
        description="Grid origin for chunk alignment",
    )

    inner_order: list[int] | None = Field(
        default=None,
        description="Dimension order for memory layout (C=row-major, F=col-major)",
    )

    read_chunk_shape: ChunkShape | None = Field(
        default=None,
        description="Chunk shape for read operations",
    )

    write_chunk_shape: ChunkShape | None = Field(
        default=None,
        description="Chunk shape for write operations",
    )

    chunk_shape: ChunkShape | None = Field(
        default=None,
        description="Chunk shape for both read and write operations",
    )

    chunk_elements: int | None = Field(
        default=None,
        description="Target number of elements per chunk",
        gt=0,
    )

    chunk_bytes: int | None = Field(
        default=None,
        description="Target size in bytes per chunk",
        gt=0,
    )

    chunk_aspect_ratio: list[float | None] | None = Field(
        default=None,
        description="Preferred aspect ratio for automatic chunk sizing",
    )

    @field_validator("inner_order", mode="before")
    @classmethod
    def validate_inner_order(cls, v: Any) -> list[int] | None:
        """Validate inner order is a valid permutation."""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("inner_order must be a list")

        if not all(isinstance(x, int) for x in v):
            raise ValueError("inner_order must contain only integers")

        # Check it's a valid permutation
        sorted_order = sorted(v)
        if sorted_order != list(range(len(v))):
            raise ValueError(
                f"inner_order must be a permutation of [0, 1, ..., {len(v) - 1}]"
            )

        return v

    @field_validator(
        "chunk_shape", "read_chunk_shape", "write_chunk_shape", mode="before"
    )
    @classmethod
    def validate_chunk_shape(cls, v: Any) -> ChunkShape | None:
        """Validate chunk shape values."""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("Chunk shape must be a list")

        for i, size in enumerate(v):
            if size is not None and size <= 0:
                raise ValueError(f"Chunk shape dimension {i} must be positive or None")

        return v

    @field_validator("chunk_aspect_ratio", mode="before")
    @classmethod
    def validate_chunk_aspect_ratio(cls, v: Any) -> list[float | None] | None:
        """Validate chunk aspect ratio values."""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("chunk_aspect_ratio must be a list")

        for i, ratio in enumerate(v):
            if ratio is not None and ratio <= 0:
                raise ValueError(
                    f"Aspect ratio for dimension {i} must be positive or None"
                )

        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        # If chunk_shape is specified, set read_chunk_shape and write_chunk_shape
        if self.chunk_shape is not None:
            if self.read_chunk_shape is None:
                self.read_chunk_shape = self.chunk_shape.copy()
            if self.write_chunk_shape is None:
                self.write_chunk_shape = self.chunk_shape.copy()

        # Validate inner_order length against chunk shapes
        if self.inner_order is not None:
            rank = len(self.inner_order)

            if self.read_chunk_shape is not None and len(self.read_chunk_shape) != rank:
                raise ValueError(
                    f"inner_order rank {rank} doesn't match "
                    f"read_chunk_shape rank {len(self.read_chunk_shape)}"
                )

            if (
                self.write_chunk_shape is not None
                and len(self.write_chunk_shape) != rank
            ):
                raise ValueError(
                    f"inner_order rank {rank} doesn't match "
                    f"write_chunk_shape rank {len(self.write_chunk_shape)}"
                )

            if self.grid_origin is not None and len(self.grid_origin) != rank:
                raise ValueError(
                    f"inner_order rank {rank} doesn't match "
                    f"grid_origin rank {len(self.grid_origin)}"
                )

    def get_effective_rank(self) -> int | None:
        """Get the effective rank from various sources."""
        if self.inner_order is not None:
            return len(self.inner_order)
        if self.read_chunk_shape is not None:
            return len(self.read_chunk_shape)
        if self.write_chunk_shape is not None:
            return len(self.write_chunk_shape)
        if self.grid_origin is not None:
            return len(self.grid_origin)
        if self.chunk_aspect_ratio is not None:
            return len(self.chunk_aspect_ratio)
        return None

    def is_c_order(self) -> bool | None:
        """Check if this represents C (row-major) order."""
        if self.inner_order is None:
            return None
        return self.inner_order == list(range(len(self.inner_order)))

    def is_f_order(self) -> bool | None:
        """Check if this represents Fortran (column-major) order."""
        if self.inner_order is None:
            return None
        return self.inner_order == list(reversed(range(len(self.inner_order))))
