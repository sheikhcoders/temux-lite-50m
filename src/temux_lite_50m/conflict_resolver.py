"""Utilities for resolving Git merge conflicts programmatically."""

from enum import Enum
from typing import Iterable, List

CONFLICT_START = "<<<<<<<"
CONFLICT_MID = "======="
CONFLICT_END = ">>>>>>>"


class ConflictResolutionStrategy(str, Enum):
    """Strategies that may be applied when resolving merge conflicts."""

    OURS = "ours"
    THEIRS = "theirs"
    COMBINE = "combine"
def has_conflict_markers(text: str) -> bool:
    """Return True if the provided text contains Git conflict markers."""

    return CONFLICT_START in text and CONFLICT_MID in text and CONFLICT_END in text


def _iter_resolved_lines(
    lines: Iterable[str], strategy: ConflictResolutionStrategy
) -> Iterable[str]:
    state = "base"
    ours: List[str] = []
    theirs: List[str] = []

    for line in lines:
        if line.startswith(CONFLICT_START):
            if state != "base":
                raise ValueError("Nested conflict markers are not supported")
            state = "ours"
            ours = []
            theirs = []
            continue

        if line.startswith(CONFLICT_MID) and state == "ours":
            state = "theirs"
            continue

        if line.startswith(CONFLICT_END) and state == "theirs":
            if strategy == ConflictResolutionStrategy.OURS:
                yield from ours
            elif strategy == ConflictResolutionStrategy.THEIRS:
                yield from theirs
            elif strategy == ConflictResolutionStrategy.COMBINE:
                yield from ours
                # ensure separation between blocks when both sides end without
                # a trailing newline
                if ours and (not ours[-1].endswith("\n") and theirs):
                    yield "\n"
                yield from theirs
            else:  # pragma: no cover - exhaustive but keeps mypy happy
                raise ValueError(f"Unknown strategy: {strategy}")
            state = "base"
            ours = []
            theirs = []
            continue

        if state == "ours":
            ours.append(line)
        elif state == "theirs":
            theirs.append(line)
        else:
            yield line

    if state != "base":
        raise ValueError("Unterminated conflict marker block detected")


def resolve_conflicts(text: str, strategy: ConflictResolutionStrategy) -> str:
    """Resolve conflicts in *text* according to *strategy*.

    Parameters
    ----------
    text:
        Raw file contents containing Git merge conflict markers.
    strategy:
        Strategy to apply when a conflict block is encountered.

    Returns
    -------
    str
        The updated file contents with conflict markers removed.
    """

    if CONFLICT_START not in text:
        return text
    if CONFLICT_MID not in text or CONFLICT_END not in text:
        raise ValueError("Unterminated conflict marker block detected")

    lines = text.splitlines(keepends=True)
    resolved_lines = list(_iter_resolved_lines(lines, strategy))
    return "".join(resolved_lines)


__all__ = [
    "ConflictResolutionStrategy",
    "has_conflict_markers",
    "resolve_conflicts",
]
