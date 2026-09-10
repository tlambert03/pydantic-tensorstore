"""Stack driver: overlays multiple TensorStores into one view."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from pydantic_tensorstore._core.spec import BaseSpec
from pydantic_tensorstore._types import ContextResource

if TYPE_CHECKING:
    from pydantic_tensorstore._drivers import TensorStoreSpec  # noqa: TC004


class StackSpec(BaseSpec):
    """Virtual view that overlays a stack of layers (later layers take precedence)."""

    driver: Literal["stack"] = "stack"
    layers: list[TensorStoreSpec | str] = Field(
        description="Layers to stack; each a TensorStore spec or URL."
    )
    data_copy_concurrency: ContextResource | None = Field(
        default=None,
        description='Concurrency for copying. Default: `"data_copy_concurrency"`.',
    )
