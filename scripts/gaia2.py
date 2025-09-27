"""Helpers for crafting Gaia2 evaluation commands."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List

DEFAULT_DATASET = "meta-agents-research-environments/gaia2"
DEFAULT_MODEL = "TheTemuxFamily/Temux-Lite-50M"
DEFAULT_PROVIDER = "local"
DEFAULT_AGENT = "default"


def build_uvx_prefix() -> List[str]:
    """Return the base uvx invocation used for Meta Agents utilities."""

    return ["uvx", "--from", "meta-agents-research-environments"]


def build_run_command(
    *,
    dataset: str = DEFAULT_DATASET,
    split: str = "validation",
    config: str,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    agent: str = DEFAULT_AGENT,
    limit: int | None = None,
    output_dir: Path | None = None,
    a2a_prop: float | None = None,
    noise: bool = False,
) -> List[str]:
    """Construct the uvx are-benchmark run command for Gaia2 scenarios."""

    if not config:
        raise ValueError("A Gaia2 configuration must be provided (e.g. 'mini', 'execution').")

    command = build_uvx_prefix() + [
        "are-benchmark",
        "run",
        "--hf-dataset",
        dataset,
        "--hf-split",
        split,
        "--hf-config",
        config,
        "--model",
        model,
        "--provider",
        provider,
        "--agent",
        agent,
    ]

    if limit is not None:
        command += ["--limit", str(limit)]

    if output_dir is not None:
        command += ["--output_dir", str(output_dir)]

    if a2a_prop is not None:
        command += ["--a2a_app_prop", str(a2a_prop)]

    if noise:
        command.append("--noise")

    return command


def build_full_evaluation_command(
    *,
    dataset: str = DEFAULT_DATASET,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    agent: str = DEFAULT_AGENT,
    output_dir: Path | None = None,
    upload_repo: str | None = None,
) -> List[str]:
    """Construct the uvx gaia2-run command that covers all configurations."""

    command = build_uvx_prefix() + [
        "are-benchmark",
        "gaia2-run",
        "--hf-dataset",
        dataset,
        "--model",
        model,
        "--provider",
        provider,
        "--agent",
        agent,
    ]

    if output_dir is not None:
        command += ["--output_dir", str(output_dir)]

    if upload_repo is not None:
        command += ["--hf_upload", upload_repo]

    return command


def build_gui_command(
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    agent: str = DEFAULT_AGENT,
) -> List[str]:
    """Construct the uvx command for launching the Meta Agents GUI."""

    return build_uvx_prefix() + [
        "are-gui",
        "-a",
        agent,
        "--model",
        model,
        "--provider",
        provider,
    ]


def run_command(args: Iterable[str]) -> int:
    """Execute the provided command list, returning the process exit code."""

    return subprocess.call(list(args))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset", default=DEFAULT_DATASET)
        subparser.add_argument("--model", default=DEFAULT_MODEL)
        subparser.add_argument("--provider", default=DEFAULT_PROVIDER)
        subparser.add_argument("--agent", default=DEFAULT_AGENT)
        subparser.add_argument("--output-dir", type=Path)
        subparser.add_argument("--print-only", action="store_true", help="Only print the command")

    validation = subparsers.add_parser("validation", help="Run a quick validation sweep")
    add_common(validation)
    validation.add_argument("--config", default="mini")
    validation.add_argument("--limit", type=int, default=20)

    capability = subparsers.add_parser("capability", help="Target a specific capability config")
    add_common(capability)
    capability.add_argument("--config", required=True)
    capability.add_argument("--limit", type=int, default=10)

    advanced = subparsers.add_parser("advanced", help="Run Agent2Agent and noise sweeps")
    add_common(advanced)
    advanced.add_argument("--config", default="mini")
    advanced.add_argument("--limit", type=int)
    advanced.add_argument("--a2a", type=float, default=1.0)
    advanced.add_argument("--noise", action="store_true")

    full = subparsers.add_parser("full", help="Run the full Gaia2 leaderboard evaluation")
    add_common(full)
    full.add_argument("--upload", help="Dataset repo to upload traces to")

    gui = subparsers.add_parser("gui", help="Launch the Meta Agents GUI")
    gui.add_argument("--model", default=DEFAULT_MODEL)
    gui.add_argument("--provider", default=DEFAULT_PROVIDER)
    gui.add_argument("--agent", default=DEFAULT_AGENT)
    gui.add_argument("--print-only", action="store_true")

    return parser.parse_args(argv)


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "validation":
        command = build_run_command(
            dataset=args.dataset,
            config=args.config,
            model=args.model,
            provider=args.provider,
            agent=args.agent,
            limit=args.limit,
            output_dir=args.output_dir,
        )
    elif args.command == "capability":
        command = build_run_command(
            dataset=args.dataset,
            config=args.config,
            model=args.model,
            provider=args.provider,
            agent=args.agent,
            limit=args.limit,
            output_dir=args.output_dir,
        )
    elif args.command == "advanced":
        command = build_run_command(
            dataset=args.dataset,
            config=args.config,
            model=args.model,
            provider=args.provider,
            agent=args.agent,
            limit=args.limit,
            output_dir=args.output_dir,
            a2a_prop=args.a2a,
            noise=args.noise,
        )
    elif args.command == "full":
        command = build_full_evaluation_command(
            dataset=args.dataset,
            model=args.model,
            provider=args.provider,
            agent=args.agent,
            output_dir=args.output_dir,
            upload_repo=args.upload,
        )
    elif args.command == "gui":
        command = build_gui_command(
            model=args.model,
            provider=args.provider,
            agent=args.agent,
        )
    else:  # pragma: no cover - argparse ensures this doesn't happen
        raise ValueError(f"Unknown command {args.command}")

    print("$", shlex.join(command))
    if getattr(args, "print_only", False):
        return 0

    if args.command == "gui" and args.print_only:
        return 0

    return run_command(command)


def main(argv: list[str] | None = None) -> int:
    return dispatch(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
