import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from source.utils import set_logger
import os
import argparse
from tqdm import tqdm
import pandas as pd
import csv
import random
import re
import json
from envs import IMG_DIR, LOG_DIR, PROMPT_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_model():
    model = (
        AutoModel.from_pretrained(
            "OpenGVLab/InternVL2_5-8B-MPO",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2_5-8B-MPO", trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


def eval_selective_alignment(method, target):
    logger = set_logger()
    logger.info(f"Start selective alignment for {method}/{target}")

    model, tokenizer = load_model()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    img_path = f"{IMG_DIR}/target_image/{method}/{target}/"
    noun_path = f"{PROMPT_DIR}/selective_alignment/{target}.json"

    # Open the file and read its contents
    with open(noun_path, 'r') as file:
        noun_data = file.read()
    nouns = json.loads(noun_data)

    responses = []
    cnt_yes = 0
    cnt_no = 0
    cnt_idk = 0
    total = len(nouns)
    with torch.no_grad():
        for noun in nouns:
            img = f"{img_path}/{noun['index']}.jpg"
            index = noun['index']

            pixel_values0 = load_image(img, max_num=12).to(torch.bfloat16).cuda()

            pixel_values = pixel_values0
            num_patches_list = [
                pixel_values0.size(0),
            ]
            prompt = noun['prompt']
            
            for noun in noun['nouns']:
                question = (
                    "You are an expert with deep knowledge in identifying unique visual concepts.\n\n"
                    "Your task:\n"
                    f"1. Look at the provided image <image> and determine if the specified {noun} is clearly visible.\n"
                    "2. If it is clearly visible, respond only with 'yes' (all lowercase, without quotes).\n"
                    "3. If it is clearly not visible, respond only with 'no'.\n"
                    "4. If you are unsure or the information is ambiguous, respond only with 'idk'.\n"
                    "5. Do not provide any explanations or additional text.\n"
                )

                response, history = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )

                last_line = response.strip()
                
                if re.search(r"\byes\b", last_line, re.IGNORECASE):
                    final_answer = "yes"
                    cnt_yes += 1
                elif re.search(r"\bidk\b", last_line, re.IGNORECASE):
                    final_answer = "idk"
                    cnt_idk += 1
                else:
                    final_answer = "no"
                    cnt_no += 1

                responses.append(
                    {"prompt": prompt, "index": index, "noun": noun, "response": response, "answer": final_answer}
                )
                print(index, prompt, noun, response, final_answer)
                print("========================================")

    log_dir = f"{LOG_DIR}/selective_alignment/{method}/{target}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert responses list to pandas DataFrame
    df_responses = pd.DataFrame(responses)
    df_responses.to_csv(f"{log_dir}/responses.csv", index=False)

    target_groups = df_responses.groupby("prompt")
    stats_rows = []
    for target_name, group in target_groups:
        total_responses = len(group)
        yes_responses = len(group[group["answer"] == "yes"])
        no_responses = len(group[group["answer"] == "no"])
        idk_responses = len(group[group["answer"] == "idk"])

        yes_rate = yes_responses / total_responses if total_responses > 0 else 0
        no_rate = no_responses / total_responses if total_responses > 0 else 0
        idk_rate = idk_responses / total_responses if total_responses > 0 else 0

        # Add row for this target
        stats_rows.append(
            {
                "target": target_name,
                "total": total_responses,
                "yes_count": yes_responses,
                "no_count": no_responses,
                "idk_count": idk_responses,
                "yes_rate": yes_rate,
                "no_rate": no_rate,
                "idk_rate": idk_rate,
            }
        )

        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(f"{log_dir}/target_stats.csv", index=False)

    logger.info(
        f"[selective_alignment/{method}/{target}]: total: {total}, yes: {cnt_yes}, no: {cnt_no}, idk: {cnt_idk}, ACC: {cnt_yes/total:.3f}"
    )

    os.makedirs(f"{LOG_DIR}/results", exist_ok=True)
    with open(f"{LOG_DIR}/results/selective_alignment.csv", "a") as f:
        f.write(
            f"{method},{target},{cnt_yes/total:.3f},{cnt_no/total:.3f},{cnt_idk/total:.3f}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="method name")
    parser.add_argument("--target", type=str, required=True, help="target concept name")
    args = parser.parse_args()
    
    eval_selective_alignment(args.method, args.target)
