"""Pytest hooks for the decision test suite."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return
    skip = pytest.mark.skip(reason="DEEPSEEK_API_KEY not set (live tests only)")
    for item in items:
        if "/live/" in item.nodeid.replace("\\", "/"):
            item.add_marker(skip)