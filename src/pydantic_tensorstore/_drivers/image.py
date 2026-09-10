"""Image drivers: single 2-D images indexed by (height, width, channel)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pydantic_tensorstore._core.spec import TensorStoreKvStoreAdapterSpec
from pydantic_tensorstore._types import DataType


class _ImageSpec(TensorStoreKvStoreAdapterSpec):
    """Common base for image drivers (uint8 only, experimental)."""

    dtype: Literal[DataType.UINT8] | None = None  # pyright: ignore[reportIncompatibleVariableOverride]


class AvifSpec(_ImageSpec):
    """AVIF image driver."""

    driver: Literal["avif"] = "avif"
    quantizer: float | None = Field(
        default=None, description="AVIF quantizer (0 lossless - 63). Default: `0`."
    )
    speed: float | None = Field(
        default=None, description="AVIF encoder speed (0-10). Default: `6`."
    )


class BmpSpec(_ImageSpec):
    """BMP image driver."""

    driver: Literal["bmp"] = "bmp"


class JpegSpec(_ImageSpec):
    """JPEG image driver."""

    driver: Literal["jpeg"] = "jpeg"
    quality: float | None = Field(
        default=None, description="JPEG quality (0-100). Default: `75`."
    )


class PngSpec(_ImageSpec):
    """PNG image driver."""

    driver: Literal["png"] = "png"
    compression_level: float | None = Field(
        default=None, description="PNG compression level (0-9)."
    )


class TiffSpec(_ImageSpec):
    """TIFF image driver (reads a very limited subset of TIFF files)."""

    driver: Literal["tiff"] = "tiff"
    page: int | None = Field(
        default=None, description="Page to read from a multi-page TIFF."
    )


class WebpSpec(_ImageSpec):
    """WebP image driver."""

    driver: Literal["webp"] = "webp"
    lossless: bool | None = Field(
        default=None, description="Lossless encoding. Default: `true`."
    )
    quality: float | None = Field(
        default=None, description="WebP quality (0-100). Default: `95`."
    )
