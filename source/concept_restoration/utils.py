import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from envs import GUIDANCE_SCALE


def get_dataset(image_dir):
    class CustomImageDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.image_paths = [
                os.path.join(image_dir, image)
                for image in os.listdir(image_dir)
                if image.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            ]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    return dataloader


@torch.no_grad()
def diffusion_step(model, latents, context, t):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (
        noise_prediction_text - noise_pred_uncond
    )

    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


@torch.no_grad()
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    raw_image = vae.decode(latents)["sample"]
    image = (raw_image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(model, height, width, generator, batch_size):
    latents = torch.randn(
        (batch_size, model.unet.in_channels, height // 8, width // 8),
        generator=generator,
        dtype=torch.float16,
    ).to(model.device)
    return latents


@torch.no_grad()
def image2latent(model, image):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(model.device)
            latents = model.vae.encode(image)["latent_dist"].mean
            latents = latents * 0.18215
    return latents


def get_context(model, prompt, device):
    target_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    target_embedding = model.text_encoder(target_input.input_ids.to(device))[0]

    # null embedding
    null_text = model.tokenizer(
        [""],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    null_embedding = model.text_encoder(null_text.input_ids.to(device))[0]

    context = torch.cat([null_embedding, target_embedding])

    return context


@torch.no_grad()
def image_restoration(image, model, prompt, start_t_idx, device):
    start_t = int(model.scheduler.timesteps[start_t_idx])

    context = get_context(model, prompt, device)

    image = (image * 2.0) - 1.0

    x0 = model.vae.encode(image)["latent_dist"].mean
    x0 = x0 * 0.18215

    noise = torch.randn_like(x0).to(device)
    xt = (
        x0 * model.scheduler.alphas_cumprod[start_t].sqrt()
        + noise * (1 - model.scheduler.alphas_cumprod[start_t]).sqrt()
    )

    for t in model.scheduler.timesteps[start_t_idx:]:
        xt = diffusion_step(model, xt, context, t)

    sampled_image = latent2image(model.vae, xt)

    return sampled_image
