import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    LMSDiscreteScheduler,
)
from diffusers.models.attention import BasicTransformerBlock

from models.stable_diffusion_reference import StableDiffusionReferencePipeline
from models.utils import load_model, diffuser_prefix_name, AttentionWithEraser
from envs import PROJECT_DIR, CACHE_DIR, SD_MODEL_NAME, CONTROLNET_NAME


def load_model_sd(
    method,
    target,
    device,
    dtype=torch.float16,
    use_controlnet=False,
    use_controlnet_reference=False,
):
    # ! Add your method in the list
    assert method in ["sd", "esd", "uce", "salun", "ac", "sa", "receler"]

    target = target.replace(" ", "_")

    if not use_controlnet and not use_controlnet_reference:
        model = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_NAME, torch_dtype=dtype, cache_dir=CACHE_DIR
        )
    elif use_controlnet:
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_NAME,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        model = StableDiffusionControlNetPipeline.from_pretrained(
            SD_MODEL_NAME,
            controlnet=controlnet,
            torch_dtype=dtype,
        )
    elif use_controlnet_reference:
        model = StableDiffusionReferencePipeline.from_pretrained(
            SD_MODEL_NAME,
            safety_checker=None,
            torch_dtype=dtype,
        )

    if method in ["esd", "uce", "salun"]:
        model.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        model.unet.load_state_dict(
            torch.load(
                f"{PROJECT_DIR}/models/sd/{method}/{target}.pt", map_location=device
            )
        )
    elif method == "ac":
        model = load_model(model, f"{PROJECT_DIR}/models/sd/ac/{target}.bin")
    elif method == "sa":
        model.unet.load_state_dict(
            torch.load(f"{PROJECT_DIR}/models/sd/sa/{target}.pt", map_location=device)
        )
    elif method == "receler":
        model.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        ckpt_path = f"{PROJECT_DIR}/models/sd/receler/{target}.pt"

        eraser_ckpt = torch.load(ckpt_path, map_location=device)

        for name, module in model.unet.named_modules():
            if isinstance(module, BasicTransformerBlock):
                prefix_name = diffuser_prefix_name(name)
                attn_w_eraser = AttentionWithEraser(module.attn2, 128)
                attn_w_eraser.adapter.load_state_dict(eraser_ckpt[prefix_name])
                module.attn2 = attn_w_eraser
        if dtype == torch.float16:
            model.unet.half()

    # ! elif method == 'YOUR_METHOD':
    # !    ...

    # To turn off NSFW filter
    def dummy(images, **kwargs):
        return images, [False]

    model.safety_checker = dummy
    model.set_progress_bar_config(disable=True)
    model = model.to(device)

    return model


class Aesthetic(nn.Module):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)
