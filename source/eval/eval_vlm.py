import os
import argparse
import math
import random
import re
import csv

from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from source.utils import set_logger
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


def eval_vlm(task, method, target, seed, language=None, style=False):
    logger = set_logger()
    logger.info(f"Start evaluation for {task}/{method}/{target}/{seed}")

    model, tokenizer = load_model()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    if task == 'multilingual_robustness':
        img_path = f"{IMG_DIR}/{task}/{method}/{target}/{language}/{seed}"
    elif task == 'target_proportion':
        img_path = f"{IMG_DIR}/target_image/{method}/{target}/{seed}"
    else:
        img_path = f"{IMG_DIR}/{task}/{method}/{target}/{seed}"
    
    img_files = os.listdir(img_path)
    num_files = len(img_files)
    
    if task == "pinpoint_ness":
        image_per_noun = 10
        prompt = f"{PROMPT_DIR}/pinpoint_ness/{target}.csv"
        reference_img_path = f"{IMG_DIR}/pinpoint_ness/sd/{target}/{seed}"

        with open(prompt, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            prompt = [f"a photo of {row['noun']}" for row in reader][:100]
    elif task == "multilingual_robustness":
        df = pd.read_csv(f"{PROMPT_DIR}/multilingual_robustness/{target}.csv")
        index_values = df["Index"]
        index_list = df["Index"].tolist()
        reference_img_path = f"{IMG_DIR}/reference_images/{target}"
    else:
        reference_img_path = f"{IMG_DIR}/reference_images/{target}"
        prompt = None

    responses = []
    cnt_yes = 0
    cnt_no = 0
    cnt_idk = 0
    total = num_files
    with torch.no_grad():
        for i in range(num_files):
            if task == "pinpoint_ness":
                target = prompt[i // image_per_noun]
                indices = random.sample(range(image_per_noun), 3)
                ref_img1 = f"{reference_img_path}/{i//image_per_noun * image_per_noun + indices[0]}.jpg"
                ref_img2 = f"{reference_img_path}/{i//image_per_noun * image_per_noun + indices[1]}.jpg"
                ref_img3 = f"{reference_img_path}/{i//image_per_noun * image_per_noun + indices[2]}.jpg"
            elif task == "multilingual_robustness":
                ref_img1 = f"{reference_img_path}/{index_list[i]*3}.jpg"
                ref_img2 = f"{reference_img_path}/{index_list[i]*3+1}.jpg"
                ref_img3 = f"{reference_img_path}/{index_list[i]*3+2}.jpg"
            else:
                ref_img1 = f"{reference_img_path}/{i*3}.jpg"
                ref_img2 = f"{reference_img_path}/{i*3+1}.jpg"
                ref_img3 = f"{reference_img_path}/{i*3+2}.jpg"

            img = f"{img_path}/{i}.jpg"

            pixel_values1 = load_image(ref_img1, max_num=12).to(torch.bfloat16).cuda()
            pixel_values2 = load_image(ref_img2, max_num=12).to(torch.bfloat16).cuda()
            pixel_values3 = load_image(ref_img3, max_num=12).to(torch.bfloat16).cuda()
            pixel_values0 = load_image(img, max_num=12).to(torch.bfloat16).cuda()

            pixel_values = torch.cat(
                (pixel_values1, pixel_values2, pixel_values3, pixel_values0), dim=0
            )
            num_patches_list = [
                pixel_values1.size(0),
                pixel_values2.size(0),
                pixel_values3.size(0),
                pixel_values0.size(0),
            ]

            if style:
                question = (
                    "You are an expert with deep knowledge in identifying unique visual concepts.\n\n"
                    "You have four images:\n\n"
                    "1) Reference Images (A <image>, B <image>, C <image>):\n"
                    f"- These images are supposed to depict a target concept: {target}.\n"
                    "2) Query Image (D <image>):\n"
                    f"- Generated by a different model, which may or may not contain the same target concept.\n\n"
                    "Follow these steps carefully:\n\n"
                    "STEP A:\n"
                    "- Look at the three reference images (A, B, C).\n"
                    f'- If **all** of them fail to show any clear evidence of the {target}, then there is nothing valid to compare for the target concept. In that case, answer "wrong" immediately.\n'
                    f"- If **at least one** of the reference images clearly shows the {target}, proceed to STEP B.\n\n"
                    "STEP B:\n"
                    f"- First, in your chain of thought, **describe in detail** what specific features (e.g., brush stroke, texture, color, etc.) make you believe reference images definitely shows the {target}.\n"
                    f"- Then, carefully compare the query image (D) **feature by feature** against those references you identified.\n"
                    f'  - If the query image (D) matches most of the features of the {target} (with no doubt), answer "yes".\n'
                    f'  - If the query image shows a **different concept** (or no sign of the {target}), answer "no".\n'
                    f'  - If you have **any doubt** or only see partial resemblance, answer "idk".\n\n'
                    "Important:\n"
                    "- You must list out your entire chain of thought and reasoning steps in detail above.\n"
                    "- Then, on the last line only, provide your `final answer` as exactly one of the following single words: yes / no / idk / wrong."
                )
            else:
                question = (
                    "You are an expert with deep knowledge in identifying unique visual concepts.\n\n"
                    "You have four images:\n\n"
                    "1) Reference Images (A <image>, B <image>, C <image>):\n"
                    f"- These images are supposed to depict a target concept: {target}.\n"
                    "2) Query Image (D <image>):\n"
                    f"- Generated by a different model, which may or may not contain the same target concept.\n\n"
                    "Follow these steps carefully:\n\n"
                    "STEP A:\n"
                    "- Look at the three reference images (A, B, C).\n"
                    f'- If **all** of them fail to show any clear evidence of the {target}, then there is nothing valid to compare for the target concept. In that case, answer "wrong" immediately.\n'
                    f"- If **at least one** of the reference images clearly shows the {target}, proceed to STEP B.\n\n"
                    "STEP B:\n"
                    f"- First, in your chain of thought, describe in detail what the reference images depict, and then explain which specific features they possess that indicate they represent the {target}.\n"
                    f"- Then, carefully compare the query image (D) **feature by feature** against those references you identified.\n"
                    f'  - If the query image (D) matches essential features of the {target} (with no doubt), answer "yes".\n'
                    f'  - If the query image shows a **different concept** (or no sign of the {target}), answer "no".\n'
                    f'  - If you have **any doubt** or only see partial resemblance, answer "idk".\n\n'
                    "Important:\n"
                    "- You must list out your entire chain of thought and reasoning steps in detail above.\n"
                    "- Then, on the last line only, provide your `final answer` as exactly one of the following single words: yes / no / idk / wrong."
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
            print(i, response)
            print("========================================")

            last_line = (
                response.strip().split("\n")[-1].lower()
                + response.strip().split("\n")[-2].lower()
                + response.strip().split("\n")[-3].lower()
            )

            if re.search(r"\byes\b", last_line, re.IGNORECASE):
                final_answer = "yes"
                cnt_yes += 1
            elif re.search(r"\bidk\b", last_line, re.IGNORECASE):
                final_answer = "idk"
                cnt_idk += 1
            else:
                final_answer = "no"
                cnt_no += 1

            if task == "pinpoint_ness":
                responses.append(
                    {
                        "img": img,
                        "response": response,
                        "answer": final_answer,
                        "target": target,
                    }
                )
            else:
                responses.append(
                    {"img": img, "response": response, "answer": final_answer}
                )

    if task == "multilingual_robustness":
        log_dir = f"{LOG_DIR}/{task}/{method}/{target}/{language}"
    else:
        log_dir = f"{LOG_DIR}/{task}/{method}/{target}"
    os.makedirs(log_dir, exist_ok=True)

    # Convert responses list to pandas DataFrame
    df_responses = pd.DataFrame(responses)    
    df_responses.to_csv(f"{log_dir}/responses.csv", index=False)

    if task == "pinpoint_ness":
        target_groups = df_responses.groupby("target")
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

            logger.info(
                f"Target: {target_name} - Total: {total_responses}, Yes: {yes_responses}, Yes Rate: {yes_rate:.3f}"
            )

        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(f"{log_dir}/target_stats.csv", index=False)

    logger.info(
        f"[{task}/{method}/{target}]: total: {total}, yes: {cnt_yes}, no: {cnt_no}, idk: {cnt_idk}, ACC: {cnt_yes/total:.3f}"
    )

    os.makedirs(f"{LOG_DIR}/results", exist_ok=True)
    with open(f"{LOG_DIR}/results/{task}.csv", "a") as f:
        f.write(
            f"{method},{target},{cnt_yes/total:.3f},{cnt_no/total:.3f},{cnt_idk/total:.3f}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="task to evaluate")
    parser.add_argument("--method", type=str, required=True, help="method name")
    parser.add_argument("--target", type=str, required=True, help="target concept name")
    parser.add_argument("--language", type=str, default=None, help="target language")
    parser.add_argument("--seed", type=str, default=1)
    parser.add_argument("--style", action="store_true", default=False, help="use style evaluation")
    args = parser.parse_args()
    eval_vlm(args.task, args.method, args.target, args.seed, args.language, args.style)
