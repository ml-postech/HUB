# Generate images from the unlearned model
import os
import sys
import csv
import json
import argparse
import math
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# Ensure the project root is importable when running as a script:
# `python source/image_generation.py ...`
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models import load_model_sd
from envs import (
    IMG_DIR,
    PROMPT_DIR,
    NUM_IMGS_PER_PROMPTS,
    LANGUAGES,
    NUM_TARGET_IMGS,
    NUM_GENERAL_IMGS,
)
from source.utils import set_seed, set_logger


def image_generation(task, method, target, seed, device, logger):
    set_seed(seed)
    logger.info(f"Start image generation for {task}/{method}/{target}/")

    # Set the number of images to generate per prompt and the prompt file
    num_per_prompt = NUM_IMGS_PER_PROMPTS[task]
    prompt = f"{PROMPT_DIR}/{task}/{target}.csv"

    # The generation of reference images for VLM evaluation should use original SD
    if task == "incontext_ref_image":
        assert method == 'sd'
        prompt = f"{PROMPT_DIR}/target_image/{target}.csv"

    # Set the prompt list
    last_prompt_num_per = None
    if task == "general_image":
        with open("prompts/MS-COCO_val2014_30k_captions.csv", "r", encoding="utf-8") as file:
            prompt = [row["text"] for row in csv.DictReader(file)][:NUM_GENERAL_IMGS]
        prompt_indices = list(range(len(prompt)))

    elif task == "multilingual_robustness":
        prompts = {lang: [] for lang in LANGUAGES}

        with open(prompt, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                for lang in LANGUAGES:
                    prompts[lang].append(row[lang])
        for lang in LANGUAGES:
            prompts[lang] = prompts[lang][:NUM_TARGET_IMGS]

    elif task == "pinpoint_ness":
        with open(prompt, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            prompt = [f"a photo of {row['noun']}" for row in reader][:100]
        # Each prompt generates `num_per_prompt` images; cap the total number of generated images
        # to `NUM_TARGET_IMGS` for small runs.
        full_prompts = len(prompt)
        max_images = full_prompts * num_per_prompt
        if NUM_TARGET_IMGS >= max_images:
            prompt_indices = list(range(full_prompts))
        else:
            full_needed_prompts = NUM_TARGET_IMGS // num_per_prompt
            remainder = NUM_TARGET_IMGS % num_per_prompt
            num_prompts = full_needed_prompts + (1 if remainder else 0)
            num_prompts = min(full_prompts, max(1, num_prompts))
            prompt_indices = list(range(num_prompts))
            if remainder and num_prompts > 0:
                last_prompt_num_per = remainder

    else:
        with open(prompt, "r", encoding="utf-8") as file:
            prompt = file.readlines()
        if task == "incontext_ref_image":
            full_count = len(prompt)
            prompt_indices = None
            multilingual_path = f"{PROMPT_DIR}/multilingual_robustness/{target}.csv"
            if os.path.isfile(multilingual_path):
                try:
                    indices = []
                    seen = set()
                    with open(multilingual_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            idx = row.get("Index")
                            if idx is None:
                                continue
                            try:
                                idx = int(idx)
                            except ValueError:
                                continue
                            if idx < 0 or idx >= full_count:
                                continue
                            if idx in seen:
                                continue
                            seen.add(idx)
                            indices.append(idx)
                            if len(indices) >= NUM_TARGET_IMGS:
                                break
                    if indices:
                        prompt_indices = indices
                except Exception as e:
                    logger.warning(
                        f"Failed to read indices from {multilingual_path}; falling back to sequential prompts. Error: {e}"
                    )

            if prompt_indices is None:
                prompt_indices = list(range(min(NUM_TARGET_IMGS, full_count)))
        if task == "target_image":
            full_count = len(prompt)
            if NUM_TARGET_IMGS >= full_count:
                prompt_indices = list(range(full_count))
            else:
                # For small runs, `prompts/selective_alignment/<target>.json` may reference
                # sparse indices (e.g. 409, 1824). Generate those indices first so
                # selective-alignment evaluation doesn't request missing images.
                selective_path = f"{PROMPT_DIR}/selective_alignment/{target}.json"
                prompt_indices = []
                if os.path.isfile(selective_path):
                    try:
                        with open(selective_path, "r", encoding="utf-8") as f:
                            nouns = json.loads(f.read())
                        raw_indices = [entry.get("index") for entry in nouns]
                        seen = set()
                        for idx in raw_indices:
                            if not isinstance(idx, int):
                                continue
                            if idx < 0 or idx >= full_count:
                                continue
                            if idx in seen:
                                continue
                            seen.add(idx)
                            prompt_indices.append(idx)
                            if len(prompt_indices) >= NUM_TARGET_IMGS:
                                break
                    except Exception as e:
                        logger.warning(
                            f"Failed to read indices from {selective_path}; falling back to sequential prompts. Error: {e}"
                        )
                        prompt_indices = []

                if len(prompt_indices) < NUM_TARGET_IMGS:
                    selected = set(prompt_indices)
                    for idx in range(full_count):
                        if idx in selected:
                            continue
                        prompt_indices.append(idx)
                        if len(prompt_indices) >= NUM_TARGET_IMGS:
                            break
        else:
            prompt_indices = list(range(len(prompt)))
            if task == "attack_robustness":
                prompt_indices = prompt_indices[: min(NUM_TARGET_IMGS, len(prompt_indices))]


    # Create a directory to save the generated images
    save_dir = f"{IMG_DIR}/{task}/{method}/{target}/{seed}"
    if task == "incontext_ref_image":
        save_dir = f"{IMG_DIR}/incontext_ref_image/{target}"
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Save images to {save_dir}")


    # Load the model
    model = load_model_sd(method, target, device)
    model.to(device)

    logger.info(f"Model loaded: {method}/{target}")


    # Generate images
    if task == "multilingual_robustness":
        for lang, lang_prompts in prompts.items():
            lang_save_dir = f"{save_dir}/{lang}"
            os.makedirs(lang_save_dir, exist_ok=True)

            for i, prompt_text in enumerate(tqdm(lang_prompts, desc=f"Generating for {lang}")):
                for j in range(num_per_prompt):
                    image = model(prompt_text, _is_progress_bar_enabled=False).images[0]
                    image.save(os.path.join(lang_save_dir, f"{i * num_per_prompt + j}.jpg"))
            
            logger.info(f"Completed generating images for {lang}")
    else:
        for i in tqdm(prompt_indices):
            per_prompt = num_per_prompt
            if task == "pinpoint_ness" and last_prompt_num_per is not None and i == prompt_indices[-1]:
                per_prompt = last_prompt_num_per
            for j in tqdm(range(per_prompt)):
                image = model(prompt[i], _is_progress_bar_enabled=False).images[0]
                image.save(f"{save_dir}/{i * num_per_prompt + j}.jpg")

    logger.info(f"Images for {task}/{method}/{target}/{seed} are generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Information of model and concept
    parser.add_argument("--method", type=str, required=True, help="method name")
    parser.add_argument("--target", type=str, required=True, help="target concept name")
    parser.add_argument(
        "--task",
        choices=[
            "target_image",
            "general_image",
            "selective_alignment",
            "pinpoint_ness",
            "multilingual_robustness",
            "attack_robustness",
            "incontext_ref_image",
        ],
        required=True,
        help="task to generate images",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    logger = set_logger()

    image_generation(args.task, args.method, args.target, args.seed, args.device, logger)
