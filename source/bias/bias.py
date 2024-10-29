import os
import csv
import yaml
import argparse

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils as tvu

from source.utils import set_seed
from source.bias.diffusion import Conditional_Model
from source.bias.lenet5 import LeNet5
from source.bias.utils import dict2namespace, get_beta_schedule, sample_image
from envs import PROJECT_DIR, LOG_DIR, IMG_DIR, MNIST_LENET_PTH


def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--method", type=str, required=True, help="Method used for unlearning"
    )
    parser.add_argument(
        "--target_class",
        type=int,
        required=True,
        help="Unlearning target class (0 ~ 9)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of images to be generated"
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Batch size for the image generation"
    )
    parser.add_argument(
        "--use_cond",
        action="store_true",
        default=False,
        help="If you want to conditionally generate the images, set it True. \
            If true, it generates images with the target_class",
    )
    args = parser.parse_args()

    return args


def bias(
    method,
    target_class,
    n_samples,
    batch_size,
    use_cond,
    seed,
    device,
):
    set_seed(seed)

    with open(f"{PROJECT_DIR}/source/bias/config.yml", "r") as fp:
        config = yaml.unsafe_load(fp)
        config = dict2namespace(config)

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)

    """
    Check your model path!
    The default path is "{PROJECT_DIR}/models/mnist/{method}/{target_class}.pth"
    """
    model_path = f"{PROJECT_DIR}/models/mnist/{method}/{target_class}.pth"
    states = torch.load(
        model_path,
        map_location=device,
    )
    model = Conditional_Model(config).to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(states[0], strict=True)
    model.eval()

    classifier = LeNet5().to(device)
    classifier.load_state_dict(torch.load(MNIST_LENET_PTH))
    classifier.eval()

    save_dir = f"{IMG_DIR}/bias/{method}/{target_class}/{seed}"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(10):
        os.makedirs(f"{save_dir}/{i}", exist_ok=True)

    classification_results = [0 for _ in range(10)]
    img_id = 0
    progress_bar = tqdm(total=n_samples)
    with torch.no_grad():
        while img_id < n_samples:
            x = torch.randn(
                batch_size,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=device,
            )
            c = torch.ones(x.size(0), device=device, dtype=int) * target_class
            x = sample_image(model, betas, x, c, use_cond).to(device)

            output = classifier(x)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)

            for k in range(batch_size):
                tvu.save_image(
                    x[k], f"{save_dir}/{predictions[k]}/{img_id}.jpg", normalize=True
                )
                classification_results[predictions[k].item()] += 1

                img_id += 1
                if img_id == n_samples:
                    break
            progress_bar.update(batch_size)

    with open(
        f"{LOG_DIR}/bias/results.csv",
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([method, target_class, seed] + classification_results)


if __name__ == "__main__":
    args = parse_args()
    bias(
        args.method,
        args.target_class,
        args.n_samples,
        args.batch_size,
        args.use_cond,
        args.seed,
        args.device,
    )
