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
AutoTokenizer = transformers.AutoTokenizer


def test_forward_pass() -> None:
    torch = pytest.importorskip("torch")
    config = AutoConfig.from_pretrained(ROOT, trust_remote_code=True)
    # Shrink the configuration so the unit test runs quickly even on CI CPUs.
    config.vocab_size = 256
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_hidden_layers = 2
    config.max_position_embeddings = 64

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.logits.shape == (2, 16, config.vocab_size)
    assert outputs.loss is not None


def test_tokenizer_roundtrip() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ROOT, trust_remote_code=True)
    encoded = tokenizer("temux on termux")
    assert "input_ids" in encoded
    assert len(encoded["input_ids"]) > 0
