"""Smoke tests for Temux-Lite-50M."""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

transformers = pytest.importorskip("transformers")
AutoConfig = transformers.AutoConfig
AutoModelForCausalLM = transformers.AutoModelForCausalLM


def test_forward_pass() -> None:
    torch = pytest.importorskip("torch")
    config = AutoConfig.from_pretrained(ROOT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.logits.shape == (2, 8, config.vocab_size)
    assert outputs.loss is not None
