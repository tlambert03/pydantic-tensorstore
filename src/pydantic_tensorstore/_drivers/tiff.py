"""TIFF driver specification for TIFF image files."""

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._core.spec import TensorStoreKvStoreAdapterSpec
from pydantic_tensorstore._types import DataType


class TiffSpec(TensorStoreKvStoreAdapterSpec):
    """TIFF driver specification for TIFF image files.

    The read volume is indexed by "height" (y), "width" (x), "channel".

    This driver is currently experimental and only supports a very limited subset of
    TIFF files. Only supports uint8 data type.
    """

    driver: Literal["tiff"] = "tiff"

    # Override dtype to restrict to uint8 only
    dtype: Literal[DataType.UINT8] | None = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default=None,
        description="Data type specification. TIFF driver only supports uint8.",
    )

    page: float | None = Field(
        default=None,
        description="Specific page number to read from multi-page TIFF files.",
    )
