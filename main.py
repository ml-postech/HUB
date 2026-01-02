import os
import argparse

from source.eval.eval_gcd import eval_with_gcd
from source.eval.eval_nsfw import eval_nsfw
from source.eval.eval_vlm import eval_vlm
from source.eval.eval_selective_alignment import eval_selective_alignment
from source.quality.evaluation import quality_evaluation
from source.utils import set_logger
from source.image_generation import image_generation
from envs import (
    IMG_DIR,
    LANGUAGES,
    STYLE_LIST,
    CELEBRITY_LIST,
    IP_LIST,
    NSFW_LIST,
    NUM_TARGET_IMGS,
    NUM_GENERAL_IMGS,
    FID_SD_IMAGE_PATH,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _count_images(dir_path):
    if not os.path.isdir(dir_path):
        return 0
    return sum(1 for f in os.listdir(dir_path) if f.lower().endswith(_IMAGE_EXTS))


def _has_any_image_recursive(dir_path):
    if not os.path.isdir(dir_path):
        return False
    for _root, _dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(_IMAGE_EXTS):
                return True
    return False


def check_images(task, method, target, seed, device, logger):
    # Check if the images exist, if not generate them
    dir_path = f"{IMG_DIR}/{task}/{method}/{target}/{seed}"
    image_count = _count_images(dir_path)
    if image_count == 0:
        logger.warning(f"Missing images in {dir_path}; generating...")
        image_generation(task, method, target, seed, device, logger)
        image_count = _count_images(dir_path)

    if task == "target_image" and image_count != NUM_TARGET_IMGS:
        logger.warning(
            f"Expected {NUM_TARGET_IMGS} images in {dir_path}, found {image_count} (continuing)."
        )
    elif task == "general_image" and image_count != NUM_GENERAL_IMGS:
        logger.warning(
            f"Expected {NUM_GENERAL_IMGS} images in {dir_path}, found {image_count} (continuing)."
        )

    return dir_path


def check_multilingual_images(method, target, seed, device, logger):
    base_dir = f"{IMG_DIR}/multilingual_robustness/{method}/{target}/{seed}"
    has_all_langs = all(_count_images(f"{base_dir}/{lang}") > 0 for lang in LANGUAGES)
    if not has_all_langs:
        logger.warning(f"Missing multilingual images under {base_dir}; generating...")
        image_generation("multilingual_robustness", method, target, seed, device, logger)


def eval(task, method, target, seed, language, device, logger):
    if task == "pinpoint_ness":
        eval_vlm(task, method, target, seed, language=None, style=(target in STYLE_LIST))
    elif target in CELEBRITY_LIST:
        eval_with_gcd(
            task, method, target, seed, language=language, device=device, logger=logger
        )
    elif target in NSFW_LIST:
        eval_nsfw(method, target, task, seed, language=language, device=device, logger=logger)
    elif target in STYLE_LIST:
        eval_vlm(task, method, target, seed, language=language, style=True)
    elif target in IP_LIST:
        eval_vlm(task, method, target, seed, language=language, style=False)
    else:
        raise ValueError(f"Unknown target concept type: {target}")


if __name__ == "__main__":
    args = get_args()
    logger = set_logger()
    logger.info(args)

    ################################################################
    # 1. Effectiveness and Faithfulness on target concept          #
    ################################################################

    # Target proportion
    logger.info(f"Target proportion")
    check_images("target_image", args.method, args.target, args.seed, args.device, logger)
    eval("target_proportion", args.method, args.target, args.seed, None, device=args.device, logger=logger)

    # General image quality
    logger.info(f"General image quality")
    check_images("general_image", args.method, args.target, args.seed, args.device, logger)
    fid_sd_ref_ready = False
    if isinstance(FID_SD_IMAGE_PATH, str):
        if FID_SD_IMAGE_PATH.endswith(".npz"):
            fid_sd_ref_ready = os.path.isfile(FID_SD_IMAGE_PATH)
        else:
            fid_sd_ref_ready = _has_any_image_recursive(FID_SD_IMAGE_PATH)

    if args.method != "sd" and not fid_sd_ref_ready:
        check_images("general_image", "sd", args.target, args.seed, args.device, logger)
    for metric in ["FID", "FID_SD"]:
        quality_evaluation(
            metric,
            "general_image",
            args.method,
            args.target,
            args.device,
            logger,
            seed=args.seed,
        )

    # Target image quality
    logger.info(f"Target image quality")
    check_images("target_image", args.method, args.target, args.seed, args.device, logger)
    quality_evaluation(
        "aesthetic",
        "target_image",
        args.method,
        args.target,
        args.device,
        logger,
        seed=args.seed,
    )


    ################################################################
    # 2. Alignment                                                 #
    ################################################################

    # General image alignment
    logger.info(f"General image alignment")
    for metric in ["ImageReward", "PickScore"]:
        quality_evaluation(
            metric,
            "general_image",
            args.method,
            args.target,
            args.device,
            logger,
            seed=args.seed,
        )

    # Target image alignment
    logger.info(f"Target image alignment")
    eval_selective_alignment(args.method, args.target, seed=args.seed)


    ################################################################
    # 3. Pinpoint_ness                                             #
    ################################################################

    logger.info(f"Pinpoint_ness")
    # Pinpoint-ness evaluation uses SD-generated reference images.
    if args.method != "sd":
        check_images("pinpoint_ness", "sd", args.target, args.seed, args.device, logger)
    check_images("pinpoint_ness", args.method, args.target, args.seed, args.device, logger)
    eval("pinpoint_ness", args.method, args.target, args.seed, None, device=args.device, logger=logger)


    ################################################################
    # 4. Multilingual robustness                                   #
    ################################################################

    logger.info(f"Multilingual robustness")
    check_multilingual_images(args.method, args.target, args.seed, args.device, logger)
    for language in LANGUAGES:
        eval(
            "multilingual_robustness",
            args.method,
            args.target,
            args.seed,
            language,
            device=args.device,
            logger=logger,
        )


    ################################################################
    # 5. Attack robustness                                         #
    ################################################################

    logger.info(f"Attack robustness")
    check_images("attack_robustness", args.method, args.target, args.seed, args.device, logger)
    eval("attack_robustness", args.method, args.target, args.seed, None, device=args.device, logger=logger)
