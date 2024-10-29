import re
import argparse
import logging
import yaml

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_logger():
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def merge_args_and_configs(config, args):
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config


def extract_concepts(sentence):
    match = re.search(r"\b(a|an) (.+?)(?:$|\s)", sentence)
    if match:
        concept = match.group(2).strip()
    else:
        assert False, f"Concept not found in sentence: {sentence}"
    return concept


def get_mean_stdinv(img):
    """
    Compute the mean and std for input image (make sure it's aligned with training)
    """

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    mean_img = np.zeros((img.shape))
    mean_img[:, :, 0] = mean[0]
    mean_img[:, :, 1] = mean[1]
    mean_img[:, :, 2] = mean[2]
    mean_img = np.float32(mean_img)

    std_img = np.zeros((img.shape))
    std_img[:, :, 0] = std[0]
    std_img[:, :, 1] = std[1]
    std_img[:, :, 2] = std[2]
    std_img = np.float64(std_img)

    stdinv_img = 1 / np.float32(std_img)

    return mean_img, stdinv_img


def numpy2tensor(img):
    """
    Convert numpy to tensor
    """
    img = torch.from_numpy(img).transpose(0, 2).transpose(1, 2).unsqueeze(0).float()
    return img


def prepare_input(img, device):
    """
    Convert numpy image into a normalized tensor (ready to do segmentation)
    """
    mean_img, stdinv_img = get_mean_stdinv(img)
    img_tensor = numpy2tensor(img).to(device)
    mean_img_tensor = numpy2tensor(mean_img).to(device)
    stdinv_img_tensor = numpy2tensor(stdinv_img).to(device)
    img_tensor = img_tensor - mean_img_tensor
    img_tensor = img_tensor * stdinv_img_tensor
    return img_tensor
