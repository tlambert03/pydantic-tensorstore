"""Stack (overlay) kvstore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from pydantic_tensorstore._core.base import TensorStoreModel
from pydantic_tensorstore._kvstore.base import BaseKvStore

if TYPE_CHECKING:
    from pydantic_tensorstore._kvstore import KvStore  # noqa: TC004


class KvStackLayer(TensorStoreModel):
    """One layer of a kvstack: a base kvstore mapped onto a key range."""

    base: KvStore = Field(description="Underlying key-value store.")
    prefix: str | None = Field(default=None, description="Key prefix this layer maps.")
    exact: str | None = Field(default=None, description="Exact key this layer maps.")
    inclusive_min: str | None = Field(default=None, description="Range lower bound.")
    exclusive_max: str | None = Field(default=None, description="Range upper bound.")
    strip_prefix: int | None = Field(
        default=None, description="Number of leading characters to strip from keys."
    )


class KvStackKvStore(BaseKvStore):
    """Stack (overlay) driver for key-value store mapping."""

    driver: Literal["kvstack"] = "kvstack"
    layers: list[KvStackLayer] = Field(description="Layer mappings, first wins.")
