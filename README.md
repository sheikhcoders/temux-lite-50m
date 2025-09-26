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
parameters with `--max-new-tokens`, `--temperature`, and `--top-p`. Add
`--no-stream` when you prefer to receive the full response in a single print
instead of token-by-token streaming.

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

## Developer kit overview
- `temux.py` — interactive CLI with chat, command, completion, and syscall
  helpers that stream output token-by-token.
- `scripts/inference.py` — single-file inference helper with argument parsing
  that mirrors the CLI defaults for quick smoke tests or demos.
- `scripts/train.py` — Hugging Face Trainer loop for continued pre-training or
  instruction tuning. Accepts local datasets or Hub datasets.
- `scripts/evaluate.py` — wraps the evaluation harness and prints an HTML-style
  report you can paste into model cards.
- `scripts/api.py` — FastAPI microservice with a `/generate` endpoint for HTTP
  integrations and VS Code extension backends.
- `scripts/benchmark.py` — latency and memory probe tailored for
  Termux/Android devices with CSV export.
- `scripts/push_to_hub.py` — guarded upload utility that keeps large artifacts
  out of git history and enforces token hygiene.
- `scripts/convert_to_safetensors.py` — helper for migrating legacy
  checkpoints.

## Inference API
Spin up a lightweight HTTP service for the model:

```bash
uvicorn scripts.api:app --host 0.0.0.0 --port 8000
```

Stream tokens back to the terminal:

```bash
curl -N -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain: ls -la", "stream": true}' \
  http://localhost:8000/generate
```

Set `"stream": false` to receive a JSON payload with `output` and `latency_ms`
fields for automation or benchmarking harnesses.

## Benchmarking toolkit
Measure end-to-end latency, throughput, and memory usage:

```bash
python scripts/benchmark.py --model TheTemuxFamily/Temux-Lite-50M --repetitions 5
```

Use `--cases custom.json` to benchmark custom workloads. The script reports the
mean tokens-per-second across runs plus the observed peak RSS in megabytes so
you can optimise for low-memory Android devices.

## Evaluation harness
The harness lives in `src/temux_lite_50m/evaluation.py` and focuses on hacker
workloads:

```bash
python scripts/evaluate.py --model TheTemuxFamily/Temux-Lite-50M
```

Results include per-case pass/fail, average latency, and tokens-per-second to
simplify benchmarking on Android/Termux devices. Custom scenarios can be loaded
by passing a JSON file through `--cases`.

## Training and fine-tuning
The provided `scripts/train.py` accepts either local datasets or public Hugging
Face Hub datasets. Example (continue pre-training on a JSONL file):

```bash
python scripts/train.py \
  --model TheTemuxFamily/Temux-Lite-50M \
  --dataset /path/to/data.jsonl \
  --output-dir outputs/temux-lite-continue \
  --per-device-train-batch-size 8 \
  --learning-rate 3e-5 \
  --num-train-epochs 3
```

Swap `--dataset` for a Hub reference such as `--dataset thetemuxfamily/cmdset`
to stream data remotely. The script saves checkpoints, TensorBoard logs, and a
`trainer_state.json` so experiments can resume mid-run.

## Tokenizer customization
`temuxlite_vocab.json` and `tokenizer_config.json` implement a lightweight
Byte-Level BPE tokenizer geared toward shell commands and code. To regenerate
the vocabulary for a new corpus, edit `tokenization_temuxlite.py` (which relies
only on the standard library) and ship the resulting files alongside the
weights. Downstream projects can also swap in a different tokenizer by pointing
`config.json`'s `auto_map` to the desired tokenizer implementation.

## Roadmap
- [ ] Publish quantized checkpoints for resource-constrained devices.
- [ ] Release a VS Code extension using `scripts/api.py` as the backend.
- [ ] Expand the evaluation harness with HumanEval/MBPP adapters.
- [ ] Add a caching layer to `temux.py` so repeated prompts reuse loaded weights.

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
- Run `git lfs install` before pushing checkpoints so large weights stay out of
  Git history.

## License
Temux-Lite-50M is distributed under the Apache License 2.0. See
[LICENSE](LICENSE) for details.
