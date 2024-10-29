import os
import datetime
import builtins
import argparse

from source.image_generation import image_generation
from source.utils.evaluation_batch import evaluation_batch
from source.quality.evaluation import quality_evaluation
from source.bias import bias
from source.concept_restoration import concept_restoration
from source.image_translation import image_translation
from source.utils import set_logger, load_config, dict2namespace, merge_args_and_configs
from envs import IMG_DIR, CONFIG_DIR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str)

    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # Load config
    config = load_config(f"{CONFIG_DIR}/{args.config}")
    config = dict2namespace(config)

    args = merge_args_and_configs(config, args)
    return args


def check_images(task, method, target, seed, device):
    # Check if the images exist, if not generate them
    dir_path = f"{IMG_DIR}/{task}/{method}/{target}/{seed}"
    try:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"
        assert len(os.listdir(dir_path)) > 0, f"No images in {dir_path}"

        if task == "simple_prompt" or task == "diverse_prompt":
            assert (
                len(os.listdir(dir_path)) == 1000
            ), f"Expected 1000 images in {dir_path}"
        elif task == "over_erasing" or task == "selective_alignment":
            assert (
                len(os.listdir(dir_path)) == 500
            ), f"Expected 500 images in {dir_path}"
        elif task == "MS-COCO":
            assert (
                len(os.listdir(dir_path)) == 30000
            ), f"Expected 30000 image in {dir_path}"

    except AssertionError as e:
        print(e)
        image_generation(task, method, target, seed, device, logger)


if __name__ == "__main__":
    args = get_args()
    logger = set_logger()
    logger.info(args)

    ################################################################
    # 1. Effectiveness and Faithfulness on target concept          #
    ################################################################

    # * Simple prompt
    if "simple_prompt" in args.tasks:
        logger.info(
            f"Evaluate images generated from simple prompt (Effectiveness and Faithfulness)"
        )
        check_images("simple_prompt", args.method, args.target, args.seed, args.device)
        evaluation_batch(
            "simple_prompt", args.method, args.target, args.seed, logger
        )  # effectivness

        quality_evaluation(
            "aesthetic",
            "simple_prompt",
            args.method,
            args.target,
            args.seed,
            args.device,
            logger,
        )

    # * Diverse prompt
    if "diverse_prompt" in args.tasks:
        logger.info(
            f"Evaluate images generated from diverse prompt (Effectiveness and Faithfulness)"
        )
        check_images("diverse_prompt", args.method, args.target, args.seed, args.device)
        evaluation_batch(
            "diverse_prompt", args.method, args.target, args.seed, logger
        )  # effectivness

        quality_evaluation(
            "aesthetic",
            "diverse_prompt",
            args.method,
            args.target,
            args.seed,
            args.device,
            logger,
        )

    ################################################################
    # 2. Faithfulness and Compliance on MS-COCO                    #
    ################################################################

    # * MS-COCO prompt
    if "MS-COCO" in args.tasks:
        logger.info(
            f"Evaluate images generated from MS-COCO prompt (Faithfulness and Compliance)"
        )
        check_images("MS-COCO", args.method, args.target, args.seed, args.device)

        quality_evaluation(
            "FID", "MS-COCO", args.method, args.target, args.seed, args.device, logger
        )
        quality_evaluation(
            "PickScore",
            "MS-COCO",
            args.method,
            args.target,
            args.seed,
            args.device,
            logger,
        )
        quality_evaluation(
            "ImageReward",
            "MS-COCO",
            args.method,
            args.target,
            args.seed,
            args.device,
            logger,
        )

    ################################################################
    # 3. Compliance on target concept                              #
    ################################################################

    # * Selective alignment
    if "selective_alignment" in args.tasks:
        logger.info(f"Evaluate selective alignment task")
        check_images(
            "selective_alignment", args.method, args.target, args.seed, args.device
        )
        evaluation_batch(
            "selective_alignment", args.method, args.target, args.seed, logger
        )

    ################################################################
    # 4. Robustness on side effects                                #
    ################################################################

    # * Over-erasing effect
    if "over_erasing" in args.tasks:
        logger.info(f"Evaluate over-erasing effect task")
        check_images("over_erasing", args.method, args.target, args.seed, args.device)
        evaluation_batch("over_erasing", args.method, args.target, args.seed, logger)

    # * Model bias
    if "bias" in args.tasks:
        logger.info(f"Evaluate performance on bias")
        bias(
            args.method,
            args.bias.target_class,
            args.bias.n_samples,
            args.bias.batch_size,
            args.bias.use_cond,
            args.seed,
            args.device,
        )

    ################################################################
    # 5. Consistency in downstream applications                    #
    ################################################################

    # * 5-1 Sketch-to-image
    if "sketch2image" in args.tasks:
        logger.info(f"Evaluate performance on sketch to image generation")
        image_translation(
            args.method, args.target, "sketch2image", args.seed, args.device
        )
        evaluation_batch("sketch2image", args.method, args.target, args.seed, logger)

    # * 5-2 Image-to-image
    if "image2image" in args.tasks:
        logger.info(f"Evaluate performance on image to image generation")
        image_translation(
            args.method, args.target, "image2image", args.seed, args.device
        )
        evaluation_batch("image2image", args.method, args.target, args.seed, logger)

    # * 5-3 Concept restoration
    if "concept_restoration" in args.tasks:
        logger.info(f"Evaluate performance on concept restoration")
        for start_t_idx in args.concept_restoration.start_t_idx:
            concept_restoration(
                args.method, args.target, start_t_idx, args.device, args.seed, logger
            )
