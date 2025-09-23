import pydantic_tensorstore


def test_imports_with_version():
    assert isinstance(pydantic_tensorstore.__version__, str)
