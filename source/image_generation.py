# Generate images from the unlearned model
import os
import csv
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from models import load_model_sd
from envs import IMG_DIR, PROMPT_DIR, NUM_IMGS_PER_PROMPTS, LANGUAGES
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
    if task == "general_image":
        with open("prompts/MS-COCO_val2014_30k_captions.csv", "r", encoding="utf-8") as file:
            prompt = [row["text"] for row in csv.DictReader(file)]

    elif task == "multilingual_robustness":
        prompts = {lang: [] for lang in LANGUAGES}

        with open(prompt, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                for lang in LANGUAGES:
                    prompts[lang].append(row[lang])

    elif task == "pinpoint_ness":
        with open(prompt, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            prompt = [f"a photo of {row['noun']}" for row in reader][:100]

    else:
        with open(prompt, "r", encoding="utf-8") as file:
            prompt = file.readlines()


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
        for i in tqdm(range(0, len(prompt))):
            for j in tqdm(range(num_per_prompt)):
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
