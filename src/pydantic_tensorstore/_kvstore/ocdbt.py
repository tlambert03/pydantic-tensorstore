"""OCDBT (Optionally-Cooperative Distributed B+Tree) kvstore adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from annotated_types import Interval
from pydantic import Field

from pydantic_tensorstore._core.base import TensorStoreModel
from pydantic_tensorstore._kvstore.base import _CachedKvStoreAdapter
from pydantic_tensorstore._types import ContextResource

if TYPE_CHECKING:
    from pydantic_tensorstore._kvstore import KvStore  # noqa: TC004


class OcdbtZstdCompression(TensorStoreModel):
    """Zstandard compression for manifest and B+tree nodes."""

    id: Literal["zstd"] = "zstd"
    level: int | None = Field(default=None, description="Compression level.")


class OcdbtConfig(TensorStoreModel):
    """Constraints on the OCDBT database configuration."""

    uuid: str | None = Field(
        default=None, description="128-bit database identifier as 32 hex digits."
    )
    manifest_kind: Literal["single", "numbered"] | None = None
    max_inline_value_bytes: Annotated[int, Interval(ge=0, le=1048576)] | None = Field(
        default=None,
        description="Max value bytes stored inline in a leaf node. Default: `100`.",
    )
    max_decoded_node_bytes: Annotated[int, Interval(ge=0, le=4294967295)] | None = (
        Field(
            default=None,
            description="Max uncompressed B+tree node size. Default: `83951616`.",
        )
    )
    version_tree_arity_log2: Annotated[int, Interval(ge=1, le=16)] | None = Field(
        default=None, description="log2 of the version tree arity. Default: `4`."
    )
    compression: OcdbtZstdCompression | None = Field(
        default=None,
        description='Node compression. Default: `{"id": "zstd", "level": 0}`.',
    )


class OcdbtKvStore(_CachedKvStoreAdapter):
    """Read/write adapter for the OCDBT format."""

    driver: Literal["ocdbt"] = "ocdbt"
    manifest: KvStore | None = Field(
        default=None, description="Separate kvstore for the manifest."
    )
    coordinator: ContextResource | None = None
    config: OcdbtConfig | None = None
    assume_config: bool | None = Field(
        default=None,
        description="Permit writing data files before the initial manifest.",
    )
    value_data_prefix: str | None = Field(
        default=None, description='Prefix for indirect value files. Default: `"d/"`.'
    )
    btree_node_data_prefix: str | None = Field(
        default=None, description='Prefix for B+tree node files. Default: `"d/"`.'
    )
    version_tree_node_data_prefix: str | None = Field(
        default=None, description='Prefix for version tree files. Default: `"d/"`.'
    )
    target_data_file_size: int | None = Field(
        default=None, description="Target data file size. Default: `2147483648`."
    )
