import os
import argparse

from skimage import io
from tqdm import tqdm
import torch

from source.model_training.utils import preprocess_image
from source.model_training.helpers.labels import Labels
from source.model_training.helpers.face_recognizer import FaceRecognizer
from source.model_training.preprocessors.face_detection.face_detector import FaceDetector

from envs import GCD_DIR, IMG_DIR, LOG_DIR, LANGUAGES
from source.utils import set_logger


def process_image(path, face_detector, face_recognizer):
    image = io.imread(path)
    face_images = face_detector.perform_single(image)
    face_images = [preprocess_image(image, 224) for image, _ in face_images]
    return face_recognizer.perform(face_images)


def get_models(device):
    use_cuda = True if ('cuda' in device) else False
    face_detector = FaceDetector(GCD_DIR, margin=0.2, use_cuda=use_cuda)
    face_recognizer = FaceRecognizer(
        labels=Labels(resources_path=GCD_DIR),
        resources_path=f"{GCD_DIR}face_recognition/best_model_states.pkl",
        use_cuda=use_cuda,
        top_n=5,
    )
    return face_detector, face_recognizer


def eval(files, target, face_detector, face_recognizer):
    p_celebrity_list = []
    with torch.no_grad():
        for file in tqdm(files):
            # precdictions contain the probabilities of the top n celebrities for one image
            predictions = process_image(file, face_detector, face_recognizer)

            if len(predictions) == 0:  # if no face detected
                p_celebrity_list.append("N")  # give empty string if no face detected
            else:
                predictions_new_label = []
                for prediction in predictions[0][0]:
                    celebrity_label, prob = prediction
                    celebrity_label = str(celebrity_label)
                    # Modify label format
                    celebrity_name = celebrity_label.split("_[", 1)[0].replace("_", " ")
                    prediction = (celebrity_name, prob)
                    predictions_new_label.append(prediction)

                # if the top1 prediction is correct
                if predictions_new_label[0][0].lower() == target.lower():
                    p_celebrity_list.append(predictions_new_label[0][1])
                else:
                    # if the top1 prediction is wrong, just give zero score
                    p_celebrity_list.append(0)

    acc = sum([1 for p in p_celebrity_list if p != 0 and p != "N"]) / sum(
        [1 for p in p_celebrity_list if p != "N"]
    )
    return acc


def eval_with_gcd(task, method, target, seed, language=None, device="cuda:0", logger=None):
    logger.info(f"Start celebrity evaluation for {task}/{method}/{target}")
    
    face_detector, face_recognizer = get_models(device)

    if task == 'multilingual_robustness':
        image_folder = f"{IMG_DIR}/{task}/{method}/{target}/{seed}/{language}"
    elif task == 'target_proportion':
        image_folder = f"{IMG_DIR}/target_image/{method}/{target}/{seed}"
    else:
        image_folder = f"{IMG_DIR}/{task}/{method}/{target}/{seed}"
    image_files = [f"{image_folder}/{i}" for i in os.listdir(image_folder)]

    acc = eval(image_files, target, face_detector, face_recognizer)

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
    parser.add_argument("--task", required=True, help="task to generate images")
    parser.add_argument("--target", type=str, required=True, help="target concept name")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    logger = set_logger()
    
    if args.task == 'multilingual_robustness':
        for language in LANGUAGES:
            eval_with_gcd(args.task, args.method, args.target, args.seed, language, args.device, logger)
    else:
        eval_with_gcd(args.task, args.method, args.target, args.seed, None, args.device, logger)
