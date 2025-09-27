"""Automatically resolve Git merge conflicts using simple heuristics."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

from temux_lite_50m.conflict_resolver import ConflictResolutionStrategy, resolve_conflicts


def get_unmerged_files(paths: Sequence[str] | None) -> List[Path]:
    if paths:
        return [Path(p) for p in paths]

    proc = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return [Path(name) for name in files]


def resolve_file(path: Path, strategy: ConflictResolutionStrategy, dry_run: bool) -> bool:
    if not path.exists():
        print(f"[skip] {path} does not exist", file=sys.stderr)
        return False

    text = path.read_text(encoding="utf-8")
    resolved = resolve_conflicts(text, strategy)
    if resolved == text:
        return False
    if dry_run:
        print(f"[dry-run] would update {path}")
    else:
        path.write_text(resolved, encoding="utf-8")
        subprocess.run(["git", "add", str(path)], check=True)
        print(f"[resolved] {path}")
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional list of files to resolve. Defaults to unmerged files.",
    )
    parser.add_argument(
        "--prefer",
        choices=[s.value for s in ConflictResolutionStrategy],
        default=ConflictResolutionStrategy.OURS.value,
        help="Conflict resolution strategy to apply (default: ours).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect conflicts without modifying files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    strategy = ConflictResolutionStrategy(args.prefer)
    files = get_unmerged_files(args.paths)

    if not files:
        print("No files to process.")
        return 0

    changed = False
    for path in files:
        try:
            changed |= resolve_file(path, strategy, args.dry_run)
        except ValueError as exc:
            print(f"[error] {path}: {exc}", file=sys.stderr)
            return 1

    if not changed:
        print("No conflict markers found in the processed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
