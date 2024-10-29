import os
import csv
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetForImageClassification
from diffusers import PNDMScheduler

from models import load_model_sd
from envs import LOG_DIR, IMG_DIR, DATASET_DIR
from source.utils import set_logger, set_seed
from source.concept_restoration.utils import get_dataset, image_restoration

target_concept_list = [
    "parachute",
    "English_springer",
    "church",
    "gas_pump",
]

labels = {
    "parachute": 701,
    "English_springer": 217,
    "church": 497,
    "gas_pump": 571,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument(
        "--target", type=str, required=True, choices=target_concept_list
    )
    parser.add_argument(
        "--start_t_idx",
        type=int,
        required=True,
        help="Start timestep of the image restoration. 0 means that concept restoration begins at t=1.0, \
            which is a noise distribution. The denoising process is divided into 50 steps between 0 and 1, \
            with each step size being 20.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    return args


def concept_restoration(method, target, start_t_idx, device, seed, logger):
    set_seed(seed)

    label = labels[target]
    prompt = f"a photo of {target.replace('_', ' ')}"
    save_path = f"{IMG_DIR}/concept_restoration/{method}/{target}/{start_t_idx}/{seed}"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/success", exist_ok=True)
    os.makedirs(f"{save_path}/fail", exist_ok=True)

    model = load_model_sd(method, target, device, dtype=torch.float32)
    model.scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        skip_prk_steps=True,
    )
    model.scheduler.set_timesteps(50)

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    classifier = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(
        device
    )
    classifier.eval()

    dataloader = get_dataset(f"{DATASET_DIR}/concept_restoration/{target}")

    start_t = int(model.scheduler.timesteps[start_t_idx])
    logger.info(f"Selected start timestep for image restoration: {start_t - 1}")

    correct = 0
    num = 0
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for idx, image in enumerate(pbar):
            image = image.to(device)
            sampled_image = image_restoration(image, model, prompt, start_t_idx, device)

            clf_inputs = processor(sampled_image, return_tensors="pt").to(device)
            clf_logits = classifier(**clf_inputs).logits
            predicted_label = clf_logits.argmax(-1).item()

            Image.fromarray(np.squeeze(sampled_image)).save(
                f"{save_path}/{'success' if predicted_label == label else 'fail'}/{idx}.jpg"
            )
            if predicted_label == label:
                correct += 1
            num += 1

            pbar.set_postfix(acc=(correct / num * 100))

    with open(
        f"{LOG_DIR}/concept_restoration/results.csv",
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                target,
                method,
                start_t,
                seed,
                correct / num * 100.0,
            ]
        )


if __name__ == "__main__":
    args = parse_args()
    logger = set_logger()
    logger.info(args)

    concept_restoration(
        method=args.method,
        target=args.target,
        start_t_idx=args.start_t_idx,
        device=args.device,
        seed=args.seed,
        logger=logger,
    )
