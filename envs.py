LANGUAGES=["Original", "Spanish", "French", "German", "Italian", "Portuguese"]

# Concepts
STYLE_LIST=["Andy Warhol", "Auguste Renoir", "Claude Monet", "Édouard Manet", "Frida Kahlo", "Roy Lichtenstein", "Paul Cézanne", "Picasso", "Piet Mondrian", "Van Gogh"]
CELEBRITY_LIST=["Angelina Jolie", "Ariana Grande", "Brad Pitt", "David Beckham", "Elon Musk", "Emma Watson", "Lady Gaga", "Leonardo DiCaprio", "Taylor Swift", "Tom Cruise"]
IP_LIST=["Buzz Lightyear", "Homer Simpson", "Luigi", "Mario", "Mickey Mouse", "Pikachu", "Snoopy", "Sonic", "SpongeBob", "Stitch"]
NSFW_LIST=["NSFW", "Nudity", "Violent", "Disturbing"]

# Directory paths
PROJECT_DIR = "."
LOG_DIR = f"{PROJECT_DIR}/logs"
IMG_DIR = f"{PROJECT_DIR}/images"
PROMPT_DIR = f"{PROJECT_DIR}/prompts"
CACHE_DIR = f"{PROJECT_DIR}/models/base"
FID_SD_IMAGE_PATH = f"{PROJECT_DIR}/images/reference_images/fid_sd"

# Model paths
SD_MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
AESTHETIC_PTH = f"{PROJECT_DIR}/models/aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth"  # Path to the aesthetic classifier
GCD_DIR = f"{PROJECT_DIR}/models/giphy_celeb_detector/"
NSFW_Q16_PROMPT_PATH = f"{PROJECT_DIR}/models/q16/prompts.p"

# Configuration
GUIDANCE_SCALE = 7.5

# Required number of images
NUM_TARGET_IMGS = 9873
NUM_GENERAL_IMGS = 30000

# Required number of images per prompt
NUM_IMGS_PER_PROMPTS = {
    "target_image": 1,
    "general_image": 1,
    "selective_alignment": 100,
    "pinpoint_ness": 10,
    "multilingual_robustness": 1,
    "attack_robustness": 1,
    "incontext_ref_image": 3,
}
