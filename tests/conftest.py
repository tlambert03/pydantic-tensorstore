from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import NoReturn

import pytest

from pydantic_tensorstore import TENSORSTORE_VERSION

PINNED_VERSION = tuple(int(x) for x in TENSORSTORE_VERSION.split("."))
try:
    TS_VERSION: tuple[int, ...] | None = tuple(
        int(x) for x in version("tensorstore").split(".")[:3]
    )
except PackageNotFoundError:
    TS_VERSION = None

# error fragments tensorstore emits for features it does not know yet
_UNSUPPORTED = (
    "is not registered",
    "Object includes extra members",
    "Unsupported data type",
    "Invalid context resource identifier",
)


def skip_if_older_tensorstore(exc: Exception) -> NoReturn:
    """Skip when an *older* tensorstore rejects a feature; re-raise otherwise.

    The pinned `TENSORSTORE_VERSION` (and anything newer) must accept everything.
    """
    if TS_VERSION is not None and TS_VERSION < PINNED_VERSION:
        if any(fragment in str(exc) for fragment in _UNSUPPORTED):
            pytest.skip(f"tensorstore {TS_VERSION} lacks this feature: {exc}")
    raise exc
