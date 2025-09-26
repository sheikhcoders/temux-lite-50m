"""Command line interface for the Temux model family."""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_MODEL_ID = "TheTemuxFamily/Temux-Lite-50M"
DEFAULT_SYSTEM_PROMPT = "You are Temux, a helpful hacker CLI assistant running inside Termux."


@dataclass
class ConversationTurn:
    role: str
    content: str


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Model repository on the Hugging Face Hub (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling value (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force loading on a specific device (cpu, cuda, mps). Defaults to auto.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Custom system prompt for chat mode.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--chat",
        action="store_true",
        help="Launch interactive chat mode.",
    )
    group.add_argument(
        "--command",
        metavar="TEXT",
        help="Explain a command or shell snippet.",
    )
    group.add_argument(
        "--complete",
        metavar="CODE",
        help="Complete a code fragment.",
    )
    group.add_argument(
        "--syscall",
        metavar="QUERY",
        help="Describe a syscall or low-level interaction.",
    )
    parser.add_argument(
        "--prompt",
        help="Optional prompt override when using completion modes.",
    )
    return parser


def load_model_components(model_id: str, device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model


def stream_generate(model, tokenizer, prompt: str, **generate_kwargs) -> Iterable[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, **generate_kwargs)
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    try:
        for token in streamer:
            yield token
    finally:
        thread.join()


def build_chat_prompt(history: List[ConversationTurn], system_prompt: str) -> str:
    messages = [f"system: {system_prompt}"]
    messages.extend(f"user: {turn.content}" if turn.role == "user" else f"assistant: {turn.content}" for turn in history)
    return "\n".join(messages) + "\nassistant:"


def interactive_chat(args, tokenizer, model):
    history: List[ConversationTurn] = []
    print("Type 'exit' or Ctrl-D to leave chat mode.")
    while True:
        try:
            user_input = input("temux> ")
        except EOFError:
            print()
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        history.append(ConversationTurn("user", user_input))
        prompt = build_chat_prompt(history, args.system)
        print("Temux:", end=" ", flush=True)
        response_tokens = []
        for token in stream_generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        ):
            response_tokens.append(token)
            print(token, end="", flush=True)
        print()
        response = "".join(response_tokens).strip()
        history.append(ConversationTurn("assistant", response))


def run_single_completion(args, tokenizer, model, mode: str, text: str) -> None:
    prompt = args.prompt or text
    if mode == "command":
        prompt = f"Explain the following shell command:\n{text}\nExplanation:"
    elif mode == "syscall":
        prompt = f"Explain the following Linux syscall in detail:\n{text}\nExplanation:"
    elif mode == "complete" and args.prompt is None:
        prompt = f"Complete the following code snippet:\n{text}\nCompletion:"
    for token in stream_generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    ):
        print(token, end="", flush=True)
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    tokenizer, model = load_model_components(args.model, args.device)
    if args.chat:
        interactive_chat(args, tokenizer, model)
    elif args.command is not None:
        run_single_completion(args, tokenizer, model, "command", args.command)
    elif args.complete is not None:
        run_single_completion(args, tokenizer, model, "complete", args.complete)
    elif args.syscall is not None:
        run_single_completion(args, tokenizer, model, "syscall", args.syscall)
    else:
        parser.error("Select one interaction mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
