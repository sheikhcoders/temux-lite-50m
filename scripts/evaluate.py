"""Run the Temux evaluation harness with an on-device or Hub model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from temux_lite_50m.evaluation import (
    DEFAULT_CASES,
    EvaluationCase,
    TemuxEvaluator,
    format_report,
)
from temux_lite_50m import ensure_model_on_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="TheTemuxFamily/Temux-Lite-50M", help="Model id or local path")
    parser.add_argument("--cases", type=Path, help="Optional path to a JSON file with custom evaluation cases")
    parser.add_argument("--max-new-tokens", type=int, default=196, help="Generation length for each case")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling value")
    parser.add_argument("--output", type=Path, help="Optional path to dump raw results as JSON")
    return parser.parse_args()


def load_cases(path: Path | None) -> List[EvaluationCase]:
    if path is None:
        return list(DEFAULT_CASES)
    with path.open("r", encoding="utf-8") as handle:
        raw_cases = json.load(handle)
    cases: List[EvaluationCase] = []
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


def make_generate_fn(model, tokenizer, args):
    def generate(prompt: str, mode: str) -> str:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text

    def tokenize(text: str):
        return tokenizer(text).input_ids

    return generate, tokenize


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    ensure_model_on_device(model, None)

    generate_fn, tokenize_fn = make_generate_fn(model, tokenizer, args)
    cases = load_cases(args.cases)
    evaluator = TemuxEvaluator(generate_fn=generate_fn, tokenizer_fn=tokenize_fn)
    results = evaluator.run(cases)
    print(format_report(results))

    if args.output:
        payload = [
            {
                "name": result.case.name,
                "success": result.success,
                "latency": result.latency,
                "tokens": result.token_count,
                "output": result.output,
            }
            for result in results
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
