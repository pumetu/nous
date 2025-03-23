import jax
import nous.nn as nn
from nous.nn import pytree_dataclass


class TransformerLayer(nn.Module):
    feedforward: nn.MLP | nn.MoE
    attn: nn.Attention
    norm_pre_attention: jax.Array
    norm_post_attention: jax.Array

    @classmethod
    def initialize(cls, config, feedforward: nn.MLP | nn.MoE):
        return cls(
            feedforward=feedforward.intialize(),
            attn=nn.Attention.initialize(),
            norm_pre_attention,
            norm_post_attention,
        )


class Transformer:
    layers: list[TransformerLayer]
    embed_tokens: jax.Array

    def initialize():
        pass

    def from_pretrained():
        pass

    def __call__():
        pass
