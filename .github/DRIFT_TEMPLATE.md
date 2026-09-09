---
title: "[drift-bot] models out of sync with latest tensorstore"
labels: [drift]
---
The weekly drift check failed: the models no longer match the newest tensorstore
release (its JSON schema files and/or runtime behaviour).

See the failed run for details: https://github.com/{{ env.GITHUB_REPOSITORY }}/actions/runs/{{ env.RUN_ID }}

To update:

1. `uv run scripts/update_ts_schema.py --latest`
2. Bump `TENSORSTORE_VERSION` in `src/pydantic_tensorstore/__init__.py`
3. `uv run pytest tests/test_schema_conformance.py` and fix the reported gaps
