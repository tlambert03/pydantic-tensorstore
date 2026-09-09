"""Fetch tensorstore's JSON-schema YAML files for a given release tag.

Usage:
    uv run scripts/update_ts_schema.py --version 0.1.85
    uv run scripts/update_ts_schema.py --latest --dest /tmp/ts_schema
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "google/tensorstore"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/{{ref}}?recursive=1"
TAGS_URL = f"https://api.github.com/repos/{REPO}/tags?per_page=20"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{{ref}}/{{path}}"
DEFAULT_DEST = Path(__file__).parent.parent / "tests" / "ts_schema"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pydantic-tensorstore"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()  # type: ignore[no-any-return]


def latest_version() -> str:
    """Return the newest `vX.Y.Z` tag of google/tensorstore."""
    tags = json.loads(_get(TAGS_URL))
    versions = [
        t["name"][1:] for t in tags if re.fullmatch(r"v\d+\.\d+\.\d+", t["name"])
    ]
    return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))


def schema_paths(ref: str) -> list[str]:
    """Return every `schema.yml` path in the repo at `ref`."""
    tree = json.loads(_get(TREE_URL.format(ref=ref)))["tree"]
    return sorted(t["path"] for t in tree if t["path"].endswith("schema.yml"))


def update(version: str, dest: Path) -> list[str]:
    """Download every schema.yml for `version` into `dest`. Returns the paths."""
    ref = f"v{version}"
    paths = schema_paths(ref)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for path in paths:
        (dest / path.replace("/", "_")).write_bytes(
            _get(RAW_URL.format(ref=ref, path=path))
        )
    (dest / "VERSION").write_text(f"{version}\n")
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", help="tensorstore version, e.g. 0.1.85")
    group.add_argument("--latest", action="store_true", help="use the newest tag")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)

    version = latest_version() if args.latest else args.version
    paths = update(version, args.dest)
    print(f"tensorstore v{version}: wrote {len(paths)} schema files to {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
