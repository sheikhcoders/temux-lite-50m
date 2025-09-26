"""Evaluation harness utilities for Temux models."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence


@dataclass
class EvaluationCase:
    """Simple task definition for the evaluation harness."""

    name: str
    prompt: str
    expected_keywords: Sequence[str]
    mode: str = "complete"


@dataclass
class EvaluationResult:
    """Result for a single evaluation case."""

    case: EvaluationCase
    output: str
    success: bool
    latency: float
    token_count: int


class TemuxEvaluator:
    """Evaluate Temux models against lightweight command-oriented tasks."""

    def __init__(
        self,
        generate_fn: Callable[[str, str], str],
        tokenizer_fn: Callable[[str], Sequence[int]] | None = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._tokenizer_fn = tokenizer_fn or (lambda text: text.split())

    def run(self, cases: Iterable[EvaluationCase]) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        for case in cases:
            start = time.perf_counter()
            output = self._generate_fn(case.prompt, case.mode)
            latency = time.perf_counter() - start
            normalized_output = output.lower()
            success = all(keyword.lower() in normalized_output for keyword in case.expected_keywords)
            token_count = len(self._tokenizer_fn(output))
            results.append(
                EvaluationResult(
                    case=case,
                    output=output,
                    success=success,
                    latency=latency,
                    token_count=token_count,
                )
            )
        return results

    @staticmethod
    def summarize(results: Sequence[EvaluationResult]) -> dict:
        if not results:
            return {"success_rate": 0.0, "avg_latency": 0.0, "tokens_per_second": 0.0}
        success_rate = sum(1 for result in results if result.success) / len(results)
        avg_latency = statistics.fmean(result.latency for result in results)
        total_tokens = sum(result.token_count for result in results)
        total_time = sum(result.latency for result in results)
        tokens_per_second = total_tokens / total_time if total_time else 0.0
        return {
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "tokens_per_second": tokens_per_second,
        }


DEFAULT_CASES: List[EvaluationCase] = [
    EvaluationCase(
        name="Explain ls -la",
        prompt="Explain what the command `ls -la` does on Linux.",
        expected_keywords=("list", "hidden", "permissions"),
        mode="command",
    ),
    EvaluationCase(
        name="Shell safety",
        prompt="Warn a user about running `rm -rf /`.",
        expected_keywords=("danger", "deletes", "root"),
        mode="command",
    ),
    EvaluationCase(
        name="Python completion",
        prompt=(
            "Complete a Python function that returns the square of a number:\n\n"
            "def square(x):"
        ),
        expected_keywords=("return", "x", "*", "x"),
        mode="complete",
    ),
]


def format_report(results: Sequence[EvaluationResult]) -> str:
    summary = TemuxEvaluator.summarize(results)
    lines = [
        "Temux Evaluation Report",
        "======================",
        f"Cases evaluated : {len(results)}",
        f"Success rate    : {summary['success_rate'] * 100:.1f}%",
        f"Avg. latency    : {summary['avg_latency'] * 1000:.1f} ms",
        f"Tokens / second : {summary['tokens_per_second']:.2f}",
        "",
    ]
    for result in results:
        status = "PASS" if result.success else "FAIL"
        lines.append(f"[{status}] {result.case.name} ({result.latency * 1000:.1f} ms)")
    return "\n".join(lines)


__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "TemuxEvaluator",
    "DEFAULT_CASES",
    "format_report",
]
