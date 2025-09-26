from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_temuxlite import TemuxLiteConfig


class Rotary(nn.Module):
    """Placeholder rotary position embedding (no-op)."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - placeholder
        return x


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act: nn.Module) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = act
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, rope_theta: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.rotary = Rotary(self.head_dim, rope_theta)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, hidden = x.size()
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.rotary(q)
        k = self.rotary(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
        attn_weights = attn_scores.softmax(dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.proj(context)


class Block(nn.Module):
    def __init__(self, config: TemuxLiteConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            rope_theta=config.rope_theta,
        )
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(
            config.hidden_size,
            config.intermediate_size,
            act=nn.GELU() if config.hidden_act == "gelu" else nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TemuxLiteModel(PreTrainedModel):
    config_class = TemuxLiteConfig

    def __init__(self, config: TemuxLiteConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)

        device = input_ids.device
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        causal_mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].eq(0)
            causal_mask = causal_mask | key_mask

        for block in self.layers:
            hidden_states = block(hidden_states, attn_mask=causal_mask)
        return self.final_layer_norm(hidden_states)


class TemuxLiteForCausalLM(PreTrainedModel):
    config_class = TemuxLiteConfig

    def __init__(self, config: TemuxLiteConfig) -> None:
        super().__init__(config)
        self.model = TemuxLiteModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutput:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        **kwargs,
    ):
        return {"input_ids": input_ids}
