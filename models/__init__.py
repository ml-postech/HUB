import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionPipeline,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.models.attention import BasicTransformerBlock

from models.sld_pipeline import SLDPipeline
from models.utils import load_model, diffuser_prefix_name, AttentionWithEraser
from envs import PROJECT_DIR, CACHE_DIR, SD_MODEL_NAME


def load_model_sd(
    method,
    target,
    device="cuda:0",
    dtype=torch.float16,
):
    # ! Add your method in the list
    assert method in ["sd", "esd", "uce", "salun", "ac", "sa", "receler", "sld", "mace"]

    if target in ["Nudity", "Disturbing", "Violent"]:
        target = "NSFW"

    if method == "sd":
        model = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_NAME, torch_dtype=dtype, cache_dir=CACHE_DIR
        )
    elif method in ["uce", "esd", "salun", "sa"]:
        model = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_NAME, torch_dtype=dtype, cache_dir=CACHE_DIR
        )
        model.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        model.unet.load_state_dict(
            torch.load(
                f"{PROJECT_DIR}/models/{method}/{target}.pt", map_location=device
            )
        )
    elif method == "ac":
        model = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_NAME, torch_dtype=dtype, cache_dir=CACHE_DIR
        )
        model = load_model(model, f"{PROJECT_DIR}/models/ac/{target}.bin")
    elif method == "receler":
        model = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_NAME, torch_dtype=dtype, cache_dir=CACHE_DIR
        )
        model.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        ckpt_path = f"{PROJECT_DIR}/models/receler/{target}.pt"

        eraser_ckpt = torch.load(ckpt_path, map_location=device)

        for name, module in model.unet.named_modules():
            if isinstance(module, BasicTransformerBlock):
                prefix_name = diffuser_prefix_name(name)
                attn_w_eraser = AttentionWithEraser(module.attn2, 128)
                attn_w_eraser.adapter.load_state_dict(eraser_ckpt[prefix_name])
                module.attn2 = attn_w_eraser
        if dtype == torch.float16:
            model.unet.half()
    elif method == "sld":
        model = SLDPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            safety_checker=None,
            torch_dtype=dtype,
        ).to(device)
        if target != "NSFW":
            model.safety_concept = target
    elif method == "mace":
        model = StableDiffusionPipeline.from_pretrained(f"{PROJECT_DIR}/models/{method}/{target}", torch_dtype=dtype).to(device)
        model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)

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
