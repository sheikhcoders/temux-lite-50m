"""Example script showing how to convert a PyTorch checkpoint to safetensors."""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to a .pt or .bin file produced by torch.save")
    parser.add_argument("output", type=str, help="Destination .safetensors file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict = torch.load(args.input, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError("Expected a state_dict dictionary in the input file")
    save_file(state_dict, args.output)


if __name__ == "__main__":
    main()
