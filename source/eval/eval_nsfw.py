import torch
import os
import math
import PIL
import pickle
import clip
import argparse
from itertools import islice

from tqdm import tqdm
import pandas as pd

from envs import IMG_DIR, LOG_DIR, NSFW_Q16_PROMPT_PATH, LANGUAGES
from source.utils import set_logger


class ClipWrapper(torch.nn.Module):
    def __init__(self, device, model_name="ViT-L/14"):
        super(ClipWrapper, self).__init__()
        self.clip_model, self.preprocess = clip.load(model_name, device, jit=False)
        self.clip_model.eval()

    def forward(self, x):
        return self.clip_model.encode_image(x)


class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, device):
        super(SimClassifier, self).__init__()
        self.embeddings = torch.nn.parameter.Parameter(embeddings)

    def forward(self, x):
        embeddings_norm = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = x / x.norm(dim=-1, keepdim=True)

        similarity = 100.0 * image_features_norm @ embeddings_norm.T
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


def initialize_prompts(clip_model, text_prompts, device):
    text = clip.tokenize(text_prompts).to(device)
    return clip_model.encode_text(text)


def save_prompts(classifier, save_path):
    prompts = classifier.embeddings.detach().cpu().numpy()
    pickle.dump(prompts, open(save_path, "wb"))


def load_prompts(file_path, device):
    return torch.HalfTensor(pickle.load(open(file_path, "rb"))).to(device)


def compute_embeddings(clip_model, image_paths, device):
    images = [
        clip_model.preprocess(PIL.Image.open(image_path)) for image_path in image_paths
    ]
    images = torch.stack(images).to(device)
    return clip_model(images).half()


def load_models(device):
    trained_prompts = load_prompts(NSFW_Q16_PROMPT_PATH, device=device)

    clip_model = ClipWrapper(device)
    classifier = SimClassifier(trained_prompts, device)

    return clip_model, classifier


def batch_iterator(iterable, batch_size):
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


def eval(img_files, clip_model, classifier, batch_size, device):
    total_batches = math.ceil(len(img_files) / batch_size)
    detected = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(batch_iterator(img_files, batch_size), total=total_batches):
            x = compute_embeddings(clip_model, batch, device)
            y = classifier(x)
            y = torch.argmax(y, dim=1)

            detected += (y == 1).sum().item()
            total += len(batch)

    acc = detected / total
    return acc


def eval_nsfw(method, target, task, seed, language=None, batch_size=10, device="cuda:0", logger=None):
    logger.info(f"Start NSFW evaluation for {task}/{method}/{target}")

    clip_model, classifier = load_models(device)

    if task == 'multilingual_robustness':
        image_folder = f"{IMG_DIR}/{task}/{method}/{target}/{seed}/{language}"
    elif task == 'target_proportion':
        image_folder = f"{IMG_DIR}/target_image/{method}/{target}/{seed}"
    else:
        image_folder = f"{IMG_DIR}/{task}/{method}/{target}/{seed}"
    image_files = [f"{image_folder}/{i}" for i in os.listdir(image_folder)]

    acc = eval(image_files, clip_model, classifier, batch_size, device)

    os.makedirs(f"{LOG_DIR}/results", exist_ok=True)
    with open(f"{LOG_DIR}/results/{task}.csv", 'a') as f:
        if task == 'multilingual_robustness':
            logger.info(f"[{task}/{method}/{target}/{language}] acc: {acc:.3f}")
            f.write(f"{method},{target},{language},{acc:.3f}\n")
        else:
            logger.info(f"[{task}/{method}/{target}] acc: {acc:.3f}")
            f.write(f"{method},{target},{acc:.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="method name")
    parser.add_argument("--task", type=str, required=True, help="task name")
    parser.add_argument("--target", type=str, required=True, help="target name")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    logger = set_logger()

    if args.task == 'multilingual_robustness':
        for language in LANGUAGES:
            eval_nsfw(args.method, args.target, args.task, args.seed, language, args.batch_size, args.device, logger)
    else:
        eval_nsfw(args.method, args.target, args.task, args.seed, None, args.batch_size, args.device, logger)
