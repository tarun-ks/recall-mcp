"""Pytest fixtures shared across the suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.make_atuin_fixture import make_atuin_fixture


@pytest.fixture(scope="session")
def atuin_fixture_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a deterministic atuin DB once per test session and return its path.

    See ``tests/fixtures/make_atuin_fixture.py`` for the row contents.
    Fixture lives under ``tmp_path`` so we never commit a binary DB.
    """
    path = tmp_path_factory.mktemp("atuin") / "history.db"
    make_atuin_fixture(path)
    return path
