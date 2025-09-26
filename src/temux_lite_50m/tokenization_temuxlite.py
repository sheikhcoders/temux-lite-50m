"""Minimal tokenizer implementation for Temux models."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional

from transformers.tokenization_utils import PreTrainedTokenizer


class TemuxLiteTokenizer(PreTrainedTokenizer):
    """A tiny, whitespace driven tokenizer.

    The tokenizer is intentionally simple so developers can fine-tune or swap in
    a project specific tokenizer later. New tokens discovered at runtime are
    appended to the vocabulary so experimentation inside notebooks or Termux
    sessions "just works" without a preprocessing step.
    """

    vocab_files_names = {"vocab_file": "temuxlite_vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        **kwargs,
    ) -> None:
        self._token_to_id: "OrderedDict[str, int]" = OrderedDict()
        if vocab is None:
            vocab = OrderedDict(
                (
                    (pad_token, 0),
                    (unk_token, 1),
                    (bos_token, 2),
                    (eos_token, 3),
                )
            )
        for token, idx in vocab.items():
            self._token_to_id[token] = idx
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}
        self.vocab = self._token_to_id
        self.ids_to_tokens = self._id_to_token
        kwargs.setdefault("unk_token", unk_token)
        kwargs.setdefault("bos_token", bos_token)
        kwargs.setdefault("eos_token", eos_token)
        kwargs.setdefault("pad_token", pad_token)
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return len(self._token_to_id)

    def get_vocab(self) -> Dict[str, int]:  # type: ignore[override]
        return dict(self._token_to_id)

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        index = len(self._token_to_id)
        self._token_to_id[token] = index
        self._id_to_token[index] = token
        return index

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return (
            [self.bos_token_id]
            + token_ids_0
            + [self.eos_token_id]
            + token_ids_1
            + [self.eos_token_id]
        )

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 1)

    def _build_conversation_prompt(self, messages: List[str]) -> str:
        return "\n".join(messages)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        os.makedirs(save_directory, exist_ok=True)
        vocab_path = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "temuxlite_vocab.json",
        )
        with open(vocab_path, "w", encoding="utf-8") as handle:
            json.dump(self.get_vocab(), handle, indent=2, ensure_ascii=False)
        return (vocab_path,)


TemuxLiteTokenizer.register_for_auto_class()


__all__ = ["TemuxLiteTokenizer"]
