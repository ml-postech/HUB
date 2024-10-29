import os
import csv
import argparse

import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import clip
import ImageReward as RM
from transformers import AutoProcessor, AutoModel
from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats

from models import Aesthetic
from envs import IMG_DIR, AESTHETIC_PTH
from source.utils import set_logger, setup_logging, log_result


def normalized(a, axis=-1, order=2):
    l2 = torch.norm(a, p=order, dim=axis, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


# * https://github.com/boomb0om/text2image-benchmark
def evaluate_fid(input_dir):
    fid, _ = calculate_fid(input_dir, get_coco_fid_stats())
    return fid


# * https://github.com/christophschuhmann/improved-aesthetic-predictor?tab=readme-ov-file
def evaluate_aesthetic(model_clip, preprocess, model, images, device):
    images = preprocess(images).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model_clip.encode_image(images)

    im_emb_arr = normalized(image_features)
    score = model(im_emb_arr.type(torch.cuda.FloatTensor)).squeeze().cpu().data.numpy()
    return score


# * https://github.com/THUDM/ImageReward
def evaluate_ImageReward(prompt, images, model):
    # * rewards = model.score("<prompt>", ["<img1_obj_or_path>", "<img2_obj_or_path>", ...])
    score = model.score(prompt, images)
    # if score is float make is list and else is just score
    if isinstance(score, float):
        score = [score]
    return score


# * https://huggingface.co/yuvalkirstain/PickScore_v1
def evaluate_pickscore(prompt, images, processor, model, device):
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

    return scores.cpu().tolist()


def quality_evaluation(metric, task, method, target, seed, device, logger):

    assert (task in ["simple_prompt", "diverse_prompt"] and metric == "aesthetic") or (
        task == "MS-COCO" and metric in ["ImageReward", "PickScore", "FID"]
    ), f"Invalid task and metric combination: {task}, {metric}"

    # set log directory
    log_file_path = setup_logging(task, method, target, seed, metric=metric)
    logger.info(f"Log file path: {log_file_path}")

    # Load MS-COCO prompts
    if task == "MS-COCO":
        with open(
            "prompts/MS-COCO_val2014_30k_captions.csv", "r", encoding="utf-8"
        ) as file:
            prompt = [row["text"] for row in csv.DictReader(file)]

    # Load the model for each metric
    if metric == "aesthetic":
        model = Aesthetic(768)
        model.load_state_dict(torch.load(AESTHETIC_PTH))
        model.eval().to(device)

        model_clip, preprocess = clip.load("ViT-L/14", device=device)
        model_clip.to(device)

    elif metric == "ImageReward":
        model = RM.load("ImageReward-v1.0")
        model.to(device)

    elif metric == "PickScore":
        processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        model = (
            AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
        )

    # Evaluate the faithfulness of generated images
    img_dir = f"{IMG_DIR}/{task}/{method}/{target}/{seed}"

    # for each image in the input directory, calculate the score
    if metric == "FID":
        score = evaluate_fid(img_dir)
        with open(log_file_path, "w") as f:
            f.write(f"FID: {score}\n")
        score_result = score

    else:
        with open(log_file_path, "w") as f:
            for i in tqdm(
                range(0, len(os.listdir(img_dir))), desc="Processing images", leave=True
            ):
                img = Image.open(f"{img_dir}/{i}.jpg")

                # Calculate the score
                if metric == "aesthetic":  # aesthetic score doesn't need prompt
                    score = evaluate_aesthetic(
                        model_clip, preprocess, model, img, device
                    )
                elif metric == "ImageReward":
                    score = evaluate_ImageReward(prompt[i], img, model)
                elif metric == "PickScore":
                    score = evaluate_pickscore(prompt[i], img, processor, model, device)

                f.write(f"{i}, {score}\n")

                if i % 100 == 0:
                    tqdm.write(f"{i} images processed")

        df = pd.read_csv(log_file_path, header=None)
        scores = df[1].tolist()
        score_result = sum(scores) / len(scores)

    log_result(score_result, task, method, target, seed, metric=metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metric",
        choices=["aesthetic", "ImageReward", "PickScore", "FID"],
        required=True,
        help="experiment type",
    )
    parser.add_argument(
        "--task",
        choices=["simple_prompt", "diverse_prompt", "MS-COCO"],
        required=True,
        help="task type",
    )

    parser.add_argument("--method", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    logger = set_logger()

    quality_evaluation(
        args.metric, args.task, args.method, args.target, args.seed, args.device, logger
    )
