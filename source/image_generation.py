# Generate images from the unlearned model
import os
import csv
import argparse
from tqdm import tqdm

from models import load_model_sd
from envs import IMG_DIR, PROMPT_DIR
from source.utils import set_seed, set_logger


def image_generation(task, method, target, seed, device, logger):
    set_seed(seed)

    logger.info(f"Start image generation for {task}/{method}/{target}/{seed}")

    # set the number of images to generate per prompt and the prompt file
    if task == "simple_prompt":
        num_per_prompt = 1000
        prompt = f"{PROMPT_DIR}/simple_prompt/{target}.csv"

    elif task == "diverse_prompt":
        num_per_prompt = 10
        prompt = f"{PROMPT_DIR}/diverse_prompt/{target}.csv"

    elif task == "over_erasing":
        num_per_prompt = 100
        prompt = f"{PROMPT_DIR}/over_erasing/{target}.csv"

    elif task == "selective_alignment":
        num_per_prompt = 100
        prompt = f"{PROMPT_DIR}/selective_alignment/{target}.csv"

    elif task == "MS-COCO":
        num_per_prompt = 1

    # Create a directory to save the generated images
    save_dir = f"{IMG_DIR}/{task}/{method}/{target}/{seed}/"
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Save images to {save_dir}")

    # Load the model
    model = load_model_sd(method, target, device)
    model.to(device)

    logger.info(f"Model loaded: {method}/{target}")

    # Set the prompt
    if task == "MS-COCO":
        with open(
            "prompts/MS-COCO_val2014_30k_captions.csv", "r", encoding="utf-8"
        ) as file:
            prompt = [row["text"] for row in csv.DictReader(file)]
    else:
        with open(prompt, "r", encoding="utf-8") as file:
            prompt = file.readlines()

    # Generate images
    for i in tqdm(range(0, len(prompt))):
        for j in range(num_per_prompt):
            image = model(prompt[i], _is_progress_bar_enabled=False).images[0]
            image.save(save_dir + f"{i * num_per_prompt + j}.jpg")
        if i % 100 == 0:
            logger.info(f"Generate {i * num_per_prompt} images")

    logger.info(f"Images for {task}/{method}/{target}/{seed} are generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Information of model and concept
    parser.add_argument("--method", type=str, required=True, help="method name")
    parser.add_argument("--target", type=str, required=True, help="target concept name")
    parser.add_argument(
        "--task",
        choices=[
            "simple_prompt",
            "diverse_prompt",
            "MS-COCO",
            "selective_alignment",
            "over_erasing",
        ],
        required=True,
        help="task to generate images",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    logger = set_logger()

    image_generation(
        args.task, args.method, args.target, args.seed, args.device, logger
    )
