"""Temux-Lite-50M public API with lazy imports to keep optional deps optional."""
from __future__ import annotations

from typing import Any

__all__ = [
    "TemuxLiteConfig",
    "TemuxLiteForCausalLM",
    "TemuxLiteModel",
    "TemuxLiteTokenizer",
    "TemuxEvaluator",
    "EvaluationCase",
    "EvaluationResult",
    "DEFAULT_CASES",
    "format_report",
    "ensure_model_on_device",
    "ConflictResolutionStrategy",
    "has_conflict_markers",
    "resolve_conflicts",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial delegation
    if name == "TemuxLiteConfig":
        from .configuration_temuxlite import TemuxLiteConfig

        return TemuxLiteConfig
    if name in {"TemuxLiteForCausalLM", "TemuxLiteModel"}:
        from .modeling_temuxlite import TemuxLiteForCausalLM, TemuxLiteModel

        return {"TemuxLiteForCausalLM": TemuxLiteForCausalLM, "TemuxLiteModel": TemuxLiteModel}[name]
    if name == "TemuxLiteTokenizer":
        from .tokenization_temuxlite import TemuxLiteTokenizer

        return TemuxLiteTokenizer
    if name in {"TemuxEvaluator", "EvaluationCase", "EvaluationResult", "DEFAULT_CASES", "format_report"}:
        from .evaluation import (
            DEFAULT_CASES,
            EvaluationCase,
            EvaluationResult,
            TemuxEvaluator,
            format_report,
        )

        return {
            "TemuxEvaluator": TemuxEvaluator,
            "EvaluationCase": EvaluationCase,
            "EvaluationResult": EvaluationResult,
            "DEFAULT_CASES": DEFAULT_CASES,
            "format_report": format_report,
        }[name]
    if name == "ensure_model_on_device":
        from .utils import ensure_model_on_device

        return ensure_model_on_device
    if name in {"ConflictResolutionStrategy", "has_conflict_markers", "resolve_conflicts"}:
        from .conflict_resolver import (
            ConflictResolutionStrategy,
            has_conflict_markers,
            resolve_conflicts,
        )

        return {
            "ConflictResolutionStrategy": ConflictResolutionStrategy,
            "has_conflict_markers": has_conflict_markers,
            "resolve_conflicts": resolve_conflicts,
        }[name]
    raise AttributeError(f"module 'temux_lite_50m' has no attribute {name!r}")
