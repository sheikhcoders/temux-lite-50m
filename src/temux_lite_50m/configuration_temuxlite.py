from transformers import PretrainedConfig


class TemuxLiteConfig(PretrainedConfig):
    """Configuration for the Temux-Lite models."""

    model_type = "temuxlite"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.tokenizer_class = "TemuxLiteTokenizer"
        self.auto_map = {
            "AutoConfig": "configuration_temuxlite.TemuxLiteConfig",
            "AutoModel": "modeling_temuxlite.TemuxLiteModel",
            "AutoModelForCausalLM": "modeling_temuxlite.TemuxLiteForCausalLM",
            "AutoTokenizer": [
                "tokenization_temuxlite.TemuxLiteTokenizer",
                None,
            ],
        }


TemuxLiteConfig.register_for_auto_class()
