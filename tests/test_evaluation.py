"""Unit tests for the Temux evaluation harness."""

from __future__ import annotations

import pathlib
import sys

import pytest

pytest.importorskip("transformers")

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from temux_lite_50m.evaluation import EvaluationCase, TemuxEvaluator, format_report


@pytest.fixture()
def dummy_cases():
    return [
        EvaluationCase(
            name="command",
            prompt="Explain ls -la",
            expected_keywords=("list", "hidden"),
            mode="command",
        ),
        EvaluationCase(
            name="completion",
            prompt="def square(x):",
            expected_keywords=("return", "x"),
            mode="complete",
        ),
    ]


def test_evaluator_success(dummy_cases):
    def generate(prompt: str, mode: str) -> str:
        if mode == "command":
            return "This will list hidden files"  # contains keywords
        return "def square(x):\n    return x * x"

    evaluator = TemuxEvaluator(generate_fn=generate)
    results = evaluator.run(dummy_cases)
    assert all(result.success for result in results)


def test_format_report_contains_status(dummy_cases):
    def generate(prompt: str, mode: str) -> str:
        return "placeholder output"

    evaluator = TemuxEvaluator(generate_fn=generate)
    results = evaluator.run(dummy_cases)
    report = format_report(results)
    assert "Temux Evaluation Report" in report
    assert "Cases evaluated" in report
