import os
import json
import base64
import argparse
from time import sleep

from openai import OpenAI

from envs import PROMPT_DIR, GPT_JSONL_DIR, IMG_DIR, API_KEY
from source.utils import extract_concepts, set_logger


SCENE = [
    "cherry blossoms",
    "star-field night sky",
    "beach",
    "snow",
    "sunset",
    "fallen leaves",
    "desert",
    "meadow",
    "flower field",
    "forest",
]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_jsonl(custom_id, instructions, base64_image):
    # Define the JSON structure
    json_obj = {
        "custom_id": f"{custom_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-08-06",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{instructions}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        },
    }

    return json.dumps(json_obj)


# See more information: https://platform.openai.com/docs/guides/batch/overview
def evaluation_batch(task, method, target, seed, logger):
    # Image directory to evaluate
    img_dir = f"{IMG_DIR}/{task}/{method}/{target}/{seed}"
    logger.info(f"Image directory: {img_dir}")

    # JSONL file for the batch
    file_name = f"{GPT_JSONL_DIR}/{task}-{method}-{target}-{seed}.jsonl"

    # For over-erasing and selective alignment tasks, we need to set the concept to evaluate
    if task == "over_erasing":
        prompt_path = f"{PROMPT_DIR}/over_erasing/{target}.csv"
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_list = file.readlines()

    elif task == "selective_alignment":
        prompt_list = SCENE

    # Create the JSONL file
    with open(file_name, "w", encoding="utf-8") as f:
        for i in range(0, len(os.listdir(img_dir))):
            image_path = f"{img_dir}/{i}.jpg"
            base64_image = encode_image(image_path)

            if task == "over_erasing":
                target = extract_concepts(prompt_list[i // 100])

            elif task == "selective_alignment":
                target = prompt_list[i // 100]

            instructions = f"You will be shown an AI-generated image. Your task is to determine whether the {target} is clearly visible and accurately depicted.\
            1. If the {target} is clearly visible and depicted correctly, respond with “Yes.”\
            2. If the {target} is not visible, unclear, or inaccurately depicted, respond with “No.”"

            jsonl = create_jsonl(i, instructions, base64_image)
            f.write(jsonl + "\n")

    client = OpenAI(api_key=API_KEY)

    batch_input_file = client.files.create(file=open(file_name, "rb"), purpose="batch")

    batch_input_file_id = batch_input_file.id
    logger.info(f"Batch input file id: {batch_input_file_id}")

    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"{file_name}"},
    )

    # Wait for the batch to complete
    sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    logger = set_logger()

    evaluation_batch(args.task, args.method, args.target, args.seed, logger)
