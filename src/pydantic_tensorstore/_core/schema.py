"""Schema model for TensorStore specifications.

The Schema defines the structure of data including data type, domain,
chunk layout, codec, fill value, and dimension units.
"""

from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from annotated_types import Interval
from pydantic import BaseModel, Field

from pydantic_tensorstore._core.chunk_layout import ChunkLayout
from pydantic_tensorstore._core.transform import IndexDomain
from pydantic_tensorstore._types import DataType, Unit

if TYPE_CHECKING:
    from pydantic_tensorstore._drivers import Codec


class Schema(BaseModel):
    """TensorStore schema specification.

    Defines the structure and constraints for TensorStore data including
    data type, domain, chunking, encoding, and physical units.
    """

    model_config: ClassVar = {"extra": "forbid", "validate_assignment": True}

    rank: Annotated[int, Interval(ge=0, le=32)] | None = Field(
        default=None, description="Number of dimensions"
    )

    dtype: DataType | None = Field(
        default=None, description="Data type of array elements"
    )

    domain: IndexDomain | None = Field(
        default=None,
        description="Domain of the TensorStore, including bounds and optional "
        "dimension labels.",
    )

    chunk_layout: ChunkLayout | None = Field(
        default=None, description="Chunk layout constraints"
    )

    codec: "Codec | None" = Field(
        default=None, description="Codec specification for compression/encoding"
    )

    fill_value: Any | None = Field(
        default=None,
        description="Fill value for unwritten elements. "
        "Must be broadcast-compatible with the domain.",
    )

    dimension_units: list[Unit | None] | None = Field(
        default=None, description="Physical units for each dimension"
    )
