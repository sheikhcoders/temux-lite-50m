"""Tests for the temux CLI argument parsing helpers."""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

temux = importlib.import_module("temux")


def test_create_parser_modes():
    parser = temux.create_parser()
    args = parser.parse_args(["--model", "local", "--command", "ls -la"])
    assert args.command == "ls -la"
    assert not args.chat


def test_no_stream_flag():
    parser = temux.create_parser()
    args = parser.parse_args(["--complete", "print('hi')", "--no-stream"])
    assert args.no_stream is True


def test_build_chat_prompt_roundtrip():
    history = [temux.ConversationTurn("user", "Hello"), temux.ConversationTurn("assistant", "Hi")]
    prompt = temux.build_chat_prompt(history, "System message")
    assert prompt.startswith("system: System message")
    assert "assistant:" in prompt
