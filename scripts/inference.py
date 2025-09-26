"""Simple inference script for Temux-Lite-50M."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "TheTemuxFamily/Temux-Lite-50M"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, trust_remote_code=True)
    model.eval()

    prompt = "Temux says: "
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
