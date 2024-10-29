import torch
import torch.nn as nn


class AttentionWithEraser(nn.Module):
    def __init__(self, attn, eraser_rank):
        super().__init__()
        self.attn = attn
        self.adapter = AdapterEraser(attn.to_out[0].weight.shape[1], eraser_rank)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ):
        attn_outputs = self.attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        return self.adapter(attn_outputs) + attn_outputs


def load_model(model, save_path):
    st = torch.load(save_path)
    if "text_encoder" in st:
        model.text_encoder.load_state_dict(st["text_encoder"])
    for name, params in model.unet.named_parameters():
        if name in st["unet"]:
            params.data.copy_(st["unet"][f"{name}"])
    return model


def diffuser_prefix_name(name):
    block_type = name.split(".")[0]
    if block_type == "mid_block":
        return ".".join(name.split(".")[:3])
    return ".".join(name.split(".")[:4])


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class EraserControlMixin:
    _use_eraser = True

    @property
    def use_eraser(self):
        return self._use_eraser

    @use_eraser.setter
    def use_eraser(self, state):
        if not isinstance(state, bool):
            raise AttributeError(f"state should be bool, but got {type(state)}.")
        self._use_eraser = state


class AdapterEraser(nn.Module, EraserControlMixin):
    def __init__(self, dim, mid_dim):
        super().__init__()
        self.down = nn.Linear(dim, mid_dim)
        self.act = nn.GELU()
        self.up = zero_module(nn.Linear(mid_dim, dim))

    def forward(self, hidden_states):
        return self.up(self.act(self.down(hidden_states)))


class AttentionWithEraser(nn.Module):
    def __init__(self, attn, eraser_rank):
        super().__init__()
        self.attn = attn
        self.adapter = AdapterEraser(attn.to_out[0].weight.shape[1], eraser_rank)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ):
        attn_outputs = self.attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        return self.adapter(attn_outputs) + attn_outputs
