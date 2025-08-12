import os
import argparse

from source.eval.eval_gcd import eval_with_gcd
from source.eval.eval_nsfw import eval_nsfw
from source.eval.eval_vlm import eval_vlm
from source.eval.eval_selective_alignment import eval_selective_alignment
from source.quality.evaluation import quality_evaluation
from source.utils import set_logger, dict2namespace
from envs import IMG_DIR, LANGUAGES, STYLE_LIST, CELEBRITY_LIST, IP_LIST, NSFW_LIST, NUM_TARGET_IMGS, NUM_GENERAL_IMGS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    return args


def check_images(task, method, target, device):
    # Check if the images exist, if not generate them
    dir_path = f"{IMG_DIR}/{task}/{method}/{target}"
    try:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"
        assert len(os.listdir(dir_path)) > 0, f"No images in {dir_path}"

        if task == "target_image":
            assert (len(os.listdir(dir_path)) == NUM_TARGET_IMGS), f"Expected 10000 images in {dir_path}"
        elif task == "general_image":
            assert (len(os.listdir(dir_path)) == NUM_GENERAL_IMGS), f"Expected 30000 images in {dir_path}"

    except AssertionError as e:
        print(e)
        image_generation(task, method, target, device, logger)


def eval(task, method, target, language, device, logger):
    if task == 'pinpoint_ness': 
        eval_vlm(task, method, target, language)
    elif target in CELEBRITY_LIST:
        eval_with_gcd(task, method, target, language,device=device, logger=logger)
    elif target in NSFW_LIST:
        eval_nsfw(task, method, target, language, device=device, logger=logger)
    elif target in STYLE_LIST:
        eval_vlm(task, method, target, language, style=True)
    elif target in IP_LIST:
        eval_vlm(task, method, target, language, style=False)


if __name__ == "__main__":
    args = get_args()
    logger = set_logger()
    logger.info(args)

    ################################################################
    # 1. Effectiveness and Faithfulness on target concept          #
    ################################################################

    # Target proportion
    logger.info(f"Target proportion")
    check_images("target_image", args.method, args.target, args.device)
    eval('target_proportion', args.method, args.target, None, device=args.device, logger=logger)

    # General image quality
    logger.info(f"General image quality")
    for metric in ["FID", "FID_SD"]:
        quality_evaluation(metric, "general_image", args.method, args.target, args.device, logger)

    # Target image quality
    logger.info(f"Target image quality")
    quality_evaluation("aesthetic", "target_image", args.method, args.target, args.device, logger)


    ################################################################
    # 2. Alignment                                                 #
    ################################################################

    # General image alignment
    logger.info(f"General image alignment")
    for metric in ["ImageReward", "PickScore"]:
        quality_evaluation(metric, "general_image", args.method, args.target, args.device, logger)

    # Target image alignment
    logger.info(f"Target image alignment")
    eval_selective_alignment(args.method, args.target)


    ################################################################
    # 3. Pinpoint_ness                                             #
    ################################################################

    logger.info(f"Pinpoint_ness")
    eval('pinpoint_ness', args.method, args.target, None, device=args.device, logger=logger)


    ################################################################
    # 4. Multilingual robustness                                   #
    ################################################################

    logger.info(f"Multilingual robustness")
    for language in LANGUAGES:
        eval('multilingual_robustness', args.method, args.target, language, device=args.device, logger=logger)


    ################################################################
    # 5. Attack robustness                                         #
    ################################################################

    logger.info(f"Attack robustness")
    eval('attack_robustness', args.method, args.target, None, device=args.device, logger=logger)
