"""Auto driver specification for automatic format detection."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import ConfigDict, Field

from pydantic_tensorstore._core.base import since
from pydantic_tensorstore._core.spec import BaseSpec
from pydantic_tensorstore._kvstore import KvStore


class AutoSpec(BaseSpec):
    """Auto driver: detects the format stored in a kvstore and delegates.

    Extra members are context resource overrides passed to the detected driver.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    driver: Literal["auto"] = Field(
        default="auto", json_schema_extra=since({"auto": "0.1.76"})
    )
    kvstore: KvStore = Field(description="Key-value store specification.")
