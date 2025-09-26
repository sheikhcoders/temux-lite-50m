# Temux-Lite-50M

Temux-Lite-50M is a compact causal language model skeleton that mirrors a Codex-style architecture. This repository provides a clean starting point for experiments, fine-tuning, and deployment on Hugging Face.

## Model summary
- Parameters: ~50M (hidden=512, layers=12, heads=8)
- Context length: 2048 tokens
- Activation: GELU
- Positional encoding: placeholder rotary module (replace with full RoPE as needed)

## Quickstart
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

repo_id = "TheTemuxFamily/Temux-Lite-50M"
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # swap to your tokenizer once trained
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

prompt = "Temux says: "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Repository layout
- `config.json`: Transformer hyperparameters for the 50M-parameter variant.
- `model_index.json`: Metadata for the Hugging Face Hub.
- `src/temux_lite_50m/`: Minimal Transformers-compatible implementation.
- `scripts/`: Utilities for inference, pushing to the Hub, and format conversion.
- `tests/`: Smoke tests to ensure the model loads with `trust_remote_code=True`.
- `weights/`: Place your `.safetensors` weights here (ignored by git).

## Setup
```bash
python -m pip install -r requirements.txt
pytest
```

## Upload to the Hub
```bash
export HUGGINGFACE_HUB_TOKEN=<your_token>
python scripts/push_to_hub.py
```

## Token hygiene
- Avoid committing API tokens to version control.
- Prefer environment variables (e.g., `HUGGINGFACE_HUB_TOKEN`).
- Rotate any tokens that may have been exposed.

## License
Temux-Lite-50M is distributed under the Apache License 2.0. See [LICENSE](LICENSE).
