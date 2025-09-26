"""Smoke tests for the FastAPI inference service."""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

api = importlib.import_module("scripts.api")


def test_create_app_has_title():
    app = api.create_app(preload=False)
    assert app.title == "Temux Inference API"
