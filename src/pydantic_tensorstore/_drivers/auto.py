"""Auto driver specification for automatic format detection."""

from typing import ClassVar, Literal

from pydantic import ConfigDict, Field

from pydantic_tensorstore._core.spec import BaseSpec
from pydantic_tensorstore._kvstore import KvStore


class AutoSpec(BaseSpec):
    """Auto driver specification for automatic format detection.

    The auto driver automatically detects the format of data stored in a key-value
    store and delegates to the appropriate TensorStore driver. This involves
    additional read requests during opening to determine the format.

    The auto driver supports chaining with other TensorStore adapters and can detect
    various TensorStore and key-value store formats.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    driver: Literal["auto"] = "auto"

    kvstore: KvStore = Field(
        description="Key-value store specification for data storage",
    )
