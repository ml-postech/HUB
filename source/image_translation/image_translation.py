import os
import argparse

import torch
from tqdm import tqdm
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector

from models import load_model_sd
from envs import IMG_DIR, DATASET_DIR
from source.utils import set_seed, set_logger
from source.utils.evaluation_batch import evaluation_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, required=True, help="Name of the method to be evaluated"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["church", "parachute", "gas_pump", "English_springer"],
        help="Target concept to be unlearned",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["sketch2image", "image2image"],
        help="Name of task. Sketch-to-image or image-to-image",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    return args


def image_translation(method, target, task, seed, device):
    set_seed(seed)
    save_dir = f"{IMG_DIR}/{task}/{method}/{target}/{seed}"
    os.makedirs(save_dir, exist_ok=True)

    if task == "sketch2image":
        hed = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        model = load_model_sd(
            method,
            target,
            device,
            dtype=torch.float32,
            use_controlnet=True,
        )
    elif task == "image2image":
        model = load_model_sd(
            method,
            target,
            device,
            dtype=torch.float32,
            use_controlnet_reference=True,
        )
        model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)

    prompt = f"a photo of {target.replace('_', ' ')}, best quality, HD, extremely detailed, realistic"
    negative_prompt = "monochrome, lowres, bad quality, bad anatomy, worst quality, low quality, low res, blurry, distortion"

    image_id = 0
    with torch.no_grad():
        pbar = tqdm(range(50))
        for i in pbar:
            image = load_image(
                f"{DATASET_DIR}/image_translation/{target}/image_{i:04d}.jpg"
            )
            for _ in range(5):
                if task == "sketch2image":
                    out_image = model(
                        prompt,
                        negative_prompt=negative_prompt,
                        image=hed(image),
                        controlnet_conditioning_scale=1.0,
                        num_inference_steps=20,
                    ).images[0]
                elif task == "image2image":
                    out_image = model(
                        prompt,
                        negative_prompt=negative_prompt,
                        ref_image=image,
                        reference_attn=True,
                        reference_adain=True,
                        num_inference_steps=20,
                    ).images[0]
                out_image.save(f"{save_dir}/{image_id}.jpg")
                image_id += 1


if __name__ == "__main__":
    args = parse_args()
    logger = set_logger()
    logger.info(args)
    image_translation(args.method, args.target, args.task, args.seed, args.device)
    evaluation_batch(args.task, args.method, args.target, args.seed, logger)
