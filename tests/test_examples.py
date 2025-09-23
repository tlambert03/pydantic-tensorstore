import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("example_file", EXAMPLES.glob("*.py"), ids=lambda p: p.name)
def test_examples(example_file: Path) -> None:
    runpy.run_path(str(example_file), run_name="__main__")
