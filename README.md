# pydantic-tensorstore

[![License](https://img.shields.io/pypi/l/pydantic-tensorstore.svg?color=green)](https://github.com/tlambert03/pydantic-tensorstore/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pydantic-tensorstore.svg?color=green)](https://pypi.org/project/pydantic-tensorstore)
[![Python Version](https://img.shields.io/pypi/pyversions/pydantic-tensorstore.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/pydantic-tensorstore/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/pydantic-tensorstore/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/pydantic-tensorstore/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/pydantic-tensorstore)

Pydantic models for the TensorStore Spec

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork tlambert03/pydantic-tensorstore --clone
# or just
# gh repo clone tlambert03/pydantic-tensorstore
cd pydantic-tensorstore
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
