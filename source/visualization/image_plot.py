import os

import matplotlib.pyplot as plt
from PIL import Image

from envs import IMG_DIR, LOG_DIR


def plot_images_by_methods(experiment, methods, concept_list, seed, image_index):
    num_rows = len(concept_list)
    num_cols = len(methods)

    if experiment == "simple" or experiment == "over-erased" or experiment == "scene":
        w = 28
        h = 4.4
    else:
        w = 28
        h = 5.5

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(w, h * num_rows))

    for row, concept in enumerate(concept_list):
        index = image_index[row]

        for col, method in enumerate(methods):
            folder_path = os.path.join(IMG_DIR, experiment, method, concept)
            seed_folders = [f for f in os.listdir(folder_path)]

            seed_folder_path = os.path.join(
                IMG_DIR, experiment, method, concept, seed_folders[seed]
            )

            img_path = os.path.join(seed_folder_path, f"{index}.jpg")
            # print(img_path)
            if os.path.exists(img_path):
                img = Image.open(img_path)

                ax = axes[row, col]
                ax.imshow(img)
                ax.axis("off")  # 축 없애기
            else:
                print(f"Image {index}.jpg not found in {seed_folder_path}")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0)
    plt.show()

    save_dir = os.path.join(LOG_DIR, experiment, "visualization")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "visualization.png"))


# 실행 예시
experiment = "over-erased"
methods = ["sd", "ac", "sa", "salun", "uce", "esd", "receler"]
concept_list = ["church", "parachute", "gas pump", "English springer"]
seed = 1
num = 10
# dex = [153, 321, 654, 997]  # 각 concept에 맞는 이미지 번호
image_index = [93, 115, 115, 199]  # 각 concept에 맞는 이미지 번호

plot_images_by_methods(experiment, methods, concept_list, seed, image_index)
