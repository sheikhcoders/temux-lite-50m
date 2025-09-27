"""Tests for the Gaia2 command helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from scripts import gaia2


def test_build_run_command_basic() -> None:
    command = gaia2.build_run_command(config="mini", limit=5, output_dir=Path("out"))
    assert command[:5] == ["uvx", "--from", "meta-agents-research-environments", "are-benchmark", "run"]
    assert "--hf-config" in command
    assert command[-2:] == ["--output_dir", "out"]


def test_build_full_evaluation_command_includes_upload() -> None:
    command = gaia2.build_full_evaluation_command(upload_repo="org/repo")
    assert "--hf_upload" in command
    assert command[-1] == "org/repo"


def test_cli_validation_print_only(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    mock_runner = mock.Mock(return_value=0)
    monkeypatch.setattr(gaia2, "run_command", mock_runner)
    exit_code = gaia2.main(["validation", "--print-only"])
    assert exit_code == 0
    mock_runner.assert_not_called()
    out = capsys.readouterr().out
    assert out.startswith("$")


def test_cli_full_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_run(cmd):
        called["cmd"] = cmd
        return 0

    monkeypatch.setattr(gaia2, "run_command", fake_run)
    exit_code = gaia2.main(["full", "--upload", "org/repo", "--output-dir", "results"])
    assert exit_code == 0
    assert called["cmd"][3:5] == ["are-benchmark", "gaia2-run"]

