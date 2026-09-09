"""Downsample driver: downsampled view of a base TensorStore."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, PositiveInt

from pydantic_tensorstore._core.spec import TensorStoreAdapterSpec

DownsampleMethod: TypeAlias = Literal["stride", "median", "mode", "mean", "min", "max"]


class DownsampleSpec(TensorStoreAdapterSpec):
    """Virtual view that downsamples the base TensorStore."""

    driver: Literal["downsample"] = "downsample"
    downsample_factors: list[PositiveInt] = Field(
        description="Downsample factor per dimension of the base TensorStore."
    )
    downsample_method: DownsampleMethod = Field(description="Downsampling method.")
