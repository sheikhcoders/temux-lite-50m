# Temux-Lite-50M

Temux-Lite-50M is the base template for the Temux family of hacker-friendly
language models. The repository packages a reproducible Transformers model
implementation, CLI tooling, and evaluation harness so every Temux variant can
share the same developer experience with different weights and vocabularies.

## Features
- **Transformers-compatible model** with a lightweight rotary-attention stack
  and configurable hyperparameters via `config.json`.
- **Custom tokenizer** implemented in pure Python for fast iteration and easy
  swaps with project-specific vocabularies.
- **Temux CLI (`temux.py`)** that supports chat, command explanation, code
  completion, and syscall descriptions with streaming output.
- **Training, inference, evaluation, and Hub utility scripts** ready for local
  or remote workflows.
- **Evaluation harness** tailored for shell and coding tasks with latency and
  throughput metrics for on-device benchmarking.

## Installation
```bash
python -m pip install -r requirements.txt
```

The scripts assume `torch`, `transformers`, and `datasets` are available. When
running inside lightweight environments such as Termux, consider installing the
CUDA/Metal builds that match your device for best performance.

## CLI quickstart
The CLI loads Temux models directly from the Hugging Face Hub and streams
responses to keep the Termux experience snappy:

```bash
python temux.py --model TheTemuxFamily/Temux-Lite-50M --chat
```

Additional modes:

```bash
# Explain a command
python temux.py --model TheTemuxFamily/Temux-Lite-50M --command "ls -la"

# Complete code from stdin or the command line
python temux.py --complete "def square(x):"

# Syscall mode for low-level exploration
python temux.py --syscall "open"
```

Use `--prompt` to override the generated prompt template, and tune sampling
parameters with `--max-new-tokens`, `--temperature`, and `--top-p`.

## Python usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TheTemuxFamily/Temux-Lite-50M", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("TheTemuxFamily/Temux-Lite-50M", trust_remote_code=True)
model.eval()

prompt = "Temux says: "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Scripts overview
- `scripts/inference.py`: quick smoke test for local weights.
- `scripts/train.py`: Hugging Face Trainer loop for continued pre-training or
  instruction tuning. Accepts local datasets or Hub datasets.
- `scripts/evaluate.py`: wraps the evaluation harness and prints a report.
- `scripts/push_to_hub.py`: uploads the repository safely without including
  large binary artifacts.
- `scripts/convert_to_safetensors.py`: helper for migrating legacy checkpoints.

## Evaluation harness
The harness lives in `src/temux_lite_50m/evaluation.py` and focuses on hacker
workloads:

```bash
python scripts/evaluate.py --model TheTemuxFamily/Temux-Lite-50M
```

Results include per-case pass/fail, average latency, and tokens-per-second to
simplify benchmarking on Android/Termux devices. Custom scenarios can be loaded
by passing a JSON file through `--cases`.

## Repository layout
```
├── temux.py                     # CLI entry point
├── config.json                  # Base model hyperparameters
├── model_index.json             # Hugging Face Hub metadata
├── src/temux_lite_50m/          # Model, tokenizer, and evaluation harness
├── scripts/                     # Developer utilities
├── tests/                       # Pytest smoke tests and harness coverage
└── weights/                     # Place `.safetensors` checkpoints here
```

## Model family roadmap
- Temux-Lite-50M: baseline 50M parameter model (this repo).
- Temux-Lite-100M: deeper variant for richer completions.
- Temux-CLI: instruction-tuned sibling optimised for terminal workflows.
- Temux-Syscall: syscall-heavy corpus for reverse engineering tasks.

Each variant can reuse this repository by swapping `config.json`, tokenizer
artifacts, and the weights stored in `weights/` or on the Hub.

## Contributing
1. Fork the repository and create a feature branch per change.
2. Run `pytest` to execute the smoke tests and evaluation harness unit tests.
3. Update documentation and examples when behaviour changes.
4. Submit a PR with clear reproduction steps and benchmark results when
   available.

Issues and feature requests are welcome—especially around tokenizer updates,
on-device optimisations, or additional evaluation scenarios.

## Token hygiene & security
- Never commit API tokens or secrets. Use environment variables like
  `HUGGINGFACE_HUB_TOKEN`.
- Rotate any tokens that may have leaked in terminals or logs.
- Use scoped tokens (read/write) with the minimum privileges required.

## License
Temux-Lite-50M is distributed under the Apache License 2.0. See
[LICENSE](LICENSE) for details.
