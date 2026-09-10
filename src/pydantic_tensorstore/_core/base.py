"""Shared base model for every pydantic-tensorstore model."""

from __future__ import annotations

import re
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_serializer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import SerializerFunctionWrapHandler

MIN_TENSORSTORE_VERSION = "0.1.68"
"""Oldest tensorstore release the models are tested against."""

_RAW_DTYPE = re.compile(r"r\d+")
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def since(version: str | dict[str, str]) -> dict[str, Any]:
    """Field metadata: the tensorstore release that introduced a field or value.

    Pass a version string for the field itself, or a `{value: version}` mapping
    for individual literal/enum values (`"r<N>"` matches zarr3 raw-bits dtypes).
    """
    return {"since": version}


def since_inline(version: str) -> dict[str, Any]:
    """Field metadata: version required only when the value is an inline object.

    Some `ContextResource` fields long predate the resource they can inline. For
    example `s3.aws_credentials` accepted a string reference in 0.1.68, but the
    inline `{"type": ...}` form only parses from 0.1.72.
    """
    return {"since_inline": version}


def parse_version(v: str) -> tuple[int, int, int]:
    """Parse the leading `X.Y.Z` of a version string."""
    if not (m := _VERSION_RE.match(v)):
        raise ValueError(f"invalid version: {v!r}")
    return int(m[1]), int(m[2]), int(m[3])


@cache
def installed_tensorstore_version() -> tuple[int, int, int] | None:
    """Version of the installed tensorstore package, or None if not installed."""
    try:
        return parse_version(version("tensorstore"))
    except PackageNotFoundError:
        return None


class UnsupportedTensorStoreVersionError(ValueError):
    """The installed tensorstore is too old for a feature used in the spec."""


class TensorStoreModel(BaseModel):
    """Base for all models.

    Serialization emits only fields that were explicitly set, plus any field whose
    value is not `None` (e.g. discriminator defaults such as `driver`). An explicit
    `None` is kept as `null`, since tensorstore gives `null` meaning for some fields
    (e.g. `compressor`, `fill_value`, `sharding`).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        # tensorstore accepts "NaN"/"Infinity"/"-Infinity" and round-trips them to
        # floats; pydantic's default of `null` would silently mean something else.
        ser_json_inf_nan="strings",
    )

    @model_serializer(mode="wrap")
    def _drop_unset_none(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        fields_set = self.model_fields_set
        for name, field in type(self).model_fields.items():
            if name in fields_set:
                continue
            for key in {name, field.alias, field.serialization_alias}:
                if key is not None and key in data and data[key] is None:
                    del data[key]
        return data

    def __repr_args__(self) -> Any:
        """Show only what was set, mirroring how the model serializes."""
        for name, value in super().__repr_args__():
            if name is None or name in self.model_fields_set or value is not None:
                yield name, value

    def version_requirements(self) -> list[tuple[str, str]]:
        """`(path, version)` for every used feature newer than the minimum version."""
        return list(_version_requirements(self))

    def required_tensorstore_version(self) -> str:
        """Oldest tensorstore release that accepts everything set on this model."""
        reqs = self.version_requirements()
        return max(
            (v for _, v in reqs), key=parse_version, default=MIN_TENSORSTORE_VERSION
        )

    def check_tensorstore_version(self) -> None:
        """Raise if the installed tensorstore is too old for this model."""
        installed = installed_tensorstore_version()
        if installed is None:
            return
        too_new = [
            (p, v)
            for p, v in self.version_requirements()
            if parse_version(v) > installed
        ]
        if too_new:
            lines = "\n".join(f"  {p}: requires tensorstore >= {v}" for p, v in too_new)
            raise UnsupportedTensorStoreVersionError(
                f"installed tensorstore {'.'.join(map(str, installed))} does not "
                f"support:\n{lines}"
            )


def _version_requirements(obj: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(obj, TensorStoreModel):
        fields = type(obj).model_fields
        for name, field in fields.items():
            value = obj.__dict__.get(name)  # not getattr: avoids deprecation warnings
            here = f"{path}.{name}" if path else name
            extra = field.json_schema_extra
            extra = extra if isinstance(extra, dict) else {}
            since_ = extra.get("since")
            if isinstance(since_, str):
                if name in obj.model_fields_set:
                    yield here, since_
            elif isinstance(since_, dict) and value is not None:
                key = str(value)
                if key not in since_ and _RAW_DTYPE.fullmatch(key):
                    key = "r<N>"
                if key in since_:
                    yield f"{here}={value!s}", str(since_[key])
            inline = extra.get("since_inline")
            if isinstance(inline, str) and isinstance(value, dict | TensorStoreModel):
                yield here, inline
            yield from _version_requirements(value, here)
        # models with extra="allow" hold undeclared members, e.g. named context
        # resources ("aws_credentials#x"); reuse the base resource's marker.
        for name, value in (obj.__pydantic_extra__ or {}).items():
            here = f"{path}.{name}" if path else name
            extra = getattr(
                fields.get(name.split("#", 1)[0]), "json_schema_extra", None
            )
            if isinstance(extra, dict):
                declared = extra.get("since")
                if isinstance(declared, str):
                    yield here, declared
                inline = extra.get("since_inline")
                if isinstance(inline, str) and isinstance(
                    value, dict | TensorStoreModel
                ):
                    yield here, inline
            yield from _version_requirements(value, here)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _version_requirements(item, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, item in obj.items():
            yield from _version_requirements(item, f"{path}.{k}")
