"""Benchmark inference speed and memory footprint for Temux models."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import fmean

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from temux_lite_50m.evaluation import DEFAULT_CASES, EvaluationCase, TemuxEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="TheTemuxFamily/Temux-Lite-50M", help="Model repo or local path")
    parser.add_argument("--cases", type=Path, help="Optional JSON file with evaluation cases")
    parser.add_argument("--device", default=None, help="Force device (cpu/cuda/mps)")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of benchmark runs to average")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before measuring")
    parser.add_argument("--max-new-tokens", type=int, default=196, help="Generation length per case")
    parser.add_argument("--output", type=Path, help="Optional path to dump raw results as JSON")
    return parser.parse_args()


def load_cases(path: Path | None) -> list[EvaluationCase]:
    if path is None:
        return list(DEFAULT_CASES)
    with path.open("r", encoding="utf-8") as handle:
        raw_cases = json.load(handle)
    cases = []
    for item in raw_cases:
        cases.append(
            EvaluationCase(
                name=item["name"],
                prompt=item["prompt"],
                expected_keywords=item.get("expected_keywords", []),
                mode=item.get("mode", "complete"),
            )
        )
    return cases


def current_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def make_generate_fn(model, tokenizer, args):
    def generate(prompt: str, mode: str) -> str:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

    def tokenize(text: str):
        return tokenizer(text).input_ids

    return generate, tokenize


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    generate_fn, tokenize_fn = make_generate_fn(model, tokenizer, args)
    cases = load_cases(args.cases)
    evaluator = TemuxEvaluator(generate_fn=generate_fn, tokenizer_fn=tokenize_fn)

    results_payload = []
    for _ in range(args.warmup):
        evaluator.run(cases)

    latencies = []
    tokens_per_second = []
    memory_peaks = []

    for _ in range(args.repetitions):
        start_mem = current_memory_mb()
        start = time.perf_counter()
        results = evaluator.run(cases)
        duration = time.perf_counter() - start
        end_mem = current_memory_mb()
        tokens = sum(result.token_count for result in results)
        latencies.append(duration)
        tokens_per_second.append(tokens / duration if duration else 0.0)
        memory_peaks.append(max(start_mem, end_mem))
        results_payload.append(
            {
                "duration_s": duration,
                "tokens": tokens,
                "tokens_per_second": tokens_per_second[-1],
                "memory_mb": memory_peaks[-1],
                "success_rate": sum(1 for r in results if r.success) / len(results),
            }
        )

    summary = {
        "mean_duration_s": fmean(latencies) if latencies else 0.0,
        "mean_tokens_per_second": fmean(tokens_per_second) if tokens_per_second else 0.0,
        "peak_memory_mb": max(memory_peaks) if memory_peaks else current_memory_mb(),
    }

    print("Temux Benchmark Summary")
    print("=======================")
    print(f"Runs                : {len(latencies)}")
    print(f"Mean duration (s)   : {summary['mean_duration_s']:.3f}")
    print(f"Mean tokens / sec   : {summary['mean_tokens_per_second']:.2f}")
    print(f"Peak memory (MB)    : {summary['peak_memory_mb']:.1f}")

    if args.output:
        payload = {"summary": summary, "runs": results_payload}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
