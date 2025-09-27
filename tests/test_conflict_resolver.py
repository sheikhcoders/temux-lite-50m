import textwrap

import pytest

from temux_lite_50m.conflict_resolver import (
    ConflictResolutionStrategy,
    has_conflict_markers,
    resolve_conflicts,
)


CONFLICT_TEXT = textwrap.dedent(
    """\
    line before
    <<<<<<< HEAD
    ours line
    =======
    theirs line
    >>>>>>> feature
    line after
    """
)


def test_has_conflict_markers():
    assert has_conflict_markers(CONFLICT_TEXT)
    assert not has_conflict_markers("no conflicts here")


@pytest.mark.parametrize(
    "strategy, expected",
    [
        (
            ConflictResolutionStrategy.OURS,
            "line before\nours line\nline after\n",
        ),
        (
            ConflictResolutionStrategy.THEIRS,
            "line before\ntheirs line\nline after\n",
        ),
        (
            ConflictResolutionStrategy.COMBINE,
            "line before\nours line\ntheirs line\nline after\n",
        ),
    ],
)
def test_resolve_conflicts(strategy, expected):
    result = resolve_conflicts(CONFLICT_TEXT, strategy)
    assert result == expected


def test_unterminated_conflict_raises():
    broken = "<<<<<<< HEAD\nours\n"  # missing separators
    with pytest.raises(ValueError):
        resolve_conflicts(broken, ConflictResolutionStrategy.OURS)
