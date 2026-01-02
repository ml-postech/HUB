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
from envs import IMG_DIR, LOG_DIR, AESTHETIC_PTH, FID_SD_IMAGE_PATH
from source.utils import set_logger


def normalized(a, axis=-1, order=2):
    l2 = torch.norm(a, p=order, dim=axis, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _list_images(dir_path):
    if not os.path.isdir(dir_path):
        return []
    return [f for f in os.listdir(dir_path) if f.lower().endswith(_IMAGE_EXTS)]


def _has_any_image_recursive(dir_path):
    if not os.path.isdir(dir_path):
        return False
    for _root, _dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(_IMAGE_EXTS):
                return True
    return False


def _numeric_stem_key(filename):
    stem, _ext = os.path.splitext(filename)
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _resolve_img_dir(task, method, target, seed):
    for task_name in (task, f"{task}_"):
        base_dir = f"{IMG_DIR}/{task_name}/{method}/{target}"
        seed_dir = f"{base_dir}/{seed}"
        if _list_images(seed_dir):
            return seed_dir
        if _list_images(base_dir):
            return base_dir
    return f"{IMG_DIR}/{task}/{method}/{target}/{seed}"


# * https://github.com/boomb0om/text2image-benchmark
def evaluate_fid(input_dir):
    fid, _ = calculate_fid(input_dir, get_coco_fid_stats())
    return fid


# * https://github.com/boomb0om/text2image-benchmark
def _resolve_fid_sd_reference(target, seed):
    if isinstance(FID_SD_IMAGE_PATH, str):
        if FID_SD_IMAGE_PATH.endswith(".npz"):
            if os.path.isfile(FID_SD_IMAGE_PATH):
                return FID_SD_IMAGE_PATH
        elif _has_any_image_recursive(FID_SD_IMAGE_PATH):
            return FID_SD_IMAGE_PATH

    fallback_dir = _resolve_img_dir("general_image", "sd", target, seed)
    if _list_images(fallback_dir):
        return fallback_dir

    raise ValueError(
        "FID_SD reference images/stats not found. "
        f"Set `FID_SD_IMAGE_PATH` in `envs.py` to either a `.npz` stats file or a directory of images, "
        f"or generate baseline SD images at '{IMG_DIR}/general_image/sd/{target}/{seed}' "
        f"(or '{IMG_DIR}/general_image_/sd/{target}/{seed}')."
    )


def evaluate_fid_sd(input_dir, target, seed):
    ref = _resolve_fid_sd_reference(target, seed)
    if os.path.abspath(str(ref)) == os.path.abspath(str(input_dir)):
        return 0.0
    fid, _ = calculate_fid(input_dir, ref)
    return fid


# * https://github.com/christophschuhmann/improved-aesthetic-predictor?tab=readme-ov-file
def evaluate_aesthetic(model_clip, preprocess, model, images, device):
    images = preprocess(images).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model_clip.encode_image(images)

    im_emb_arr = normalized(image_features)
    score = model(im_emb_arr.float()).squeeze().detach().cpu().data.numpy()

    if isinstance(score, list):
        score = score[0]

    return score


# * https://github.com/THUDM/ImageReward
def evaluate_ImageReward(prompt, images, model):
    # * rewards = model.score("<prompt>", ["<img1_obj_or_path>", "<img2_obj_or_path>", ...])
    score = model.score(prompt, images)
    # if score is float make is list and else is just score
    # if isinstance(score, float):
    #     score = [score]
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


def quality_evaluation(metric, task, method, target, device, logger, seed=1):

    assert (task == "target_image" and metric == "aesthetic") or (
        task == "general_image" and metric in ["ImageReward", "PickScore", "FID", "FID_SD"]
    ), f"Invalid task and metric combination: {task}, {metric}"

    # Load MS-COCO prompts
    if metric in ["ImageReward", "PickScore"]:
        with open(
            "prompts/MS-COCO_val2014_30k_captions.csv", "r", encoding="utf-8"
        ) as file:
            prompt = [row["text"] for row in csv.DictReader(file)]

    # Load the model for each metric
    if metric == "aesthetic":
        if not os.path.isfile(AESTHETIC_PTH):
            raise FileNotFoundError(
                f"Aesthetic weights not found at '{AESTHETIC_PTH}'. "
                "Download `sac+logos+ava1-l14-linearMSE.pth` and place it there."
            )
        with open(AESTHETIC_PTH, "rb") as f:
            header = f.read(256).lstrip()
        if header.startswith(b"<!DOCTYPE html") or header.startswith(b"<html"):
            raise ValueError(
                f"'{AESTHETIC_PTH}' looks like an HTML page (common when downloading from GitHub without the raw URL). "
                "Re-download the weights file, e.g.\n"
                "wget -O models/aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth "
                "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
            )
        model = Aesthetic(768)
        model.load_state_dict(torch.load(AESTHETIC_PTH, map_location="cpu"))
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
    img_dir = _resolve_img_dir(task, method, target, seed)
    image_files = sorted(_list_images(img_dir), key=_numeric_stem_key)
    if not image_files:
        raise ValueError(
            f"No images found for {task}/{method}/{target} (seed={seed}). "
            f"Expected images under '{IMG_DIR}/{task}/{method}/{target}/{seed}'. "
            f"If you generated into '{IMG_DIR}/{task}_/...', either rename that folder "
            f"or regenerate with: python source/image_generation.py --method {method} "
            f'--target "{target}" --task {task} --seed {seed} --device {device}'
        )

    # for each image in the input directory, calculate the score
    if metric == "FID":
        score_result = evaluate_fid(img_dir)

    # for each image in the input directory, calculate the score
    elif metric == "FID_SD":
        score_result = evaluate_fid_sd(img_dir, target, seed)

    else:
        scores = 0.0
        for filename in tqdm(image_files):
            img = Image.open(f"{img_dir}/{filename}")
            key_kind, key_value = _numeric_stem_key(filename)
            idx = key_value if key_kind == 0 else None

            # Calculate the score
            if metric == "aesthetic":  # aesthetic score doesn't need prompt
                score = evaluate_aesthetic(model_clip, preprocess, model, img, device)
            elif metric == "ImageReward":
                if idx is None:
                    raise ValueError(
                        f"Unexpected non-numeric filename for {metric}: {filename}"
                    )
                score = evaluate_ImageReward(prompt[idx], img, model)
            elif metric == "PickScore":
                if idx is None:
                    raise ValueError(
                        f"Unexpected non-numeric filename for {metric}: {filename}"
                    )
                score = evaluate_pickscore(prompt[idx], img, processor, model, device)

            if isinstance(score, list):
                score = score[0]
            scores += float(score)
        score_result = scores / len(image_files)


    if task == "general_image" and metric in ["FID", "FID_SD"]:
        os.makedirs(f"{LOG_DIR}/results", exist_ok=True)
        with open(f"{LOG_DIR}/results/general_image_quality.csv", "a") as f:
            f.write(f"{method},{target},{score_result},{metric}\n")
    elif task == "general_image" and metric in ["ImageReward", "PickScore"]:
        os.makedirs(f"{LOG_DIR}/results", exist_ok=True)
        with open(f"{LOG_DIR}/results/general_image_alignment.csv", "a") as f:
            f.write(f"{method},{target},{score_result},{metric}\n")
    elif task == "target_image" and metric == "aesthetic":
        os.makedirs(f"{LOG_DIR}/results", exist_ok=True)
        with open(f"{LOG_DIR}/results/target_image_quality.csv", "a") as f:
            f.write(f"{method},{target},{score_result},{metric}\n")
    else:
        raise ValueError(f"Invalid task settings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metric",
        choices=["aesthetic", "ImageReward", "PickScore", "FID", "FID_SD"],
        required=True,
        help="experiment type",
    )
    parser.add_argument(
        "--task",
        choices=["general_image", "target_image"],
        required=True,
        help="task type",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    logger = set_logger()

    quality_evaluation(
        args.metric,
        args.task,
        args.method,
        args.target,
        args.device,
        logger,
        seed=args.seed,
    )
