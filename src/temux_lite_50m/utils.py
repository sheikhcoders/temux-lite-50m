"""Utility helpers shared across Temux tooling."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

DeviceLike = Optional[Union[str, torch.device]]


def ensure_model_on_device(model: nn.Module, device: DeviceLike) -> torch.device:
    """Move ``model`` to ``device`` and coerce precision for CPU inference.

    Transformers configs often default to ``float16`` weights for GPU
    deployment, but CPUs lack LayerNorm kernels in half precision. To make the
    developer tooling "just work" out-of-the-box, this helper promotes the
    parameters to ``float32`` whenever the resolved device is CPU.
    """

    resolved = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(resolved)
    if resolved.type == "cpu":
        model.to(dtype=torch.float32)
    return resolved


__all__ = ["ensure_model_on_device"]

