# API key for gpt-4o
API_KEY = ...

# Directory paths
PROJECT_DIR = "./"
LOG_DIR = f"{PROJECT_DIR}/logs"
IMG_DIR = f"{PROJECT_DIR}/images"
CONFIG_DIR = f"{PROJECT_DIR}/configs"
PROMPT_DIR = f"{PROJECT_DIR}/prompts"
DATASET_DIR = f"{PROJECT_DIR}/datasets"
CACHE_DIR = f"{PROJECT_DIR}/models/base"
GPT_JSONL_DIR = f"{PROJECT_DIR}/gpt_jsonl"

# Model paths
SD_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
CONTROLNET_NAME = "lllyasviel/control_v11p_sd15_softedge"
AESTHETIC_PTH = f"{PROJECT_DIR}/models/aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth"  # Path to the aesthetic classifier
MNIST_LENET_PTH = f"{PROJECT_DIR}/models/mnist/lenet5.pt"

# Configuration
GUIDANCE_SCALE = 7.5
