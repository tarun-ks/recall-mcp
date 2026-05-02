import recall


def test_version_set() -> None:
    assert isinstance(recall.__version__, str)
    assert recall.__version__
