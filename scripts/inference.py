"""Command-line inference helper for Temux models."""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Iterable, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

DEFAULT_MODEL = "TheTemuxFamily/Temux-Lite-50M"


def stream_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Iterable[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    worker = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    worker.start()
    try:
        for token in streamer:
            yield token
    finally:
        worker.join()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Prompt to feed the model. Reads stdin when omitted.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model repository ID (default: %(default)s)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to sample (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Force-load model on a device.")
    return parser


def resolve_prompt(parsed: argparse.Namespace) -> str:
    if parsed.prompt:
        return parsed.prompt
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    raise SystemExit("Provide a prompt argument or pipe non-empty text via stdin.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    prompt = resolve_prompt(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    for token in stream_generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
