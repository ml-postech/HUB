import os

from envs import LOG_DIR


def setup_logging(task, method, target, seed, metric=None):
    if metric is None:
        if task == "over_erasing" or task == "selective_alignment":
            log_dir = os.path.join(LOG_DIR, task, method, target)
        else:  # effectiveness: evaluation using gpt-4o
            log_dir = os.path.join(LOG_DIR, "effectiveness", task, method, target)

    elif (
        task == "simple_prompt" or task == "diverse_prompt" or task == "MS-COCO"
    ):  # faithfulness / metric: aesthetic, PickScore, FID, ImageReward
        log_dir = os.path.join(LOG_DIR, "faithfulness", task, metric, method, target)

    else:
        raise ValueError(f"Invalid task: {task}")

    os.makedirs(log_dir, exist_ok=True)

    # set log file path
    log_file_path = os.path.join(log_dir, f"{seed}.csv")

    return log_file_path


# log the final result
def log_result(score, task, method, target, seed, metric=None):
    os.makedirs(f"{LOG_DIR}/results/", exist_ok=True)

    # set log file path
    if metric is None:  # task that using gpt-4o
        if task == "over_erasing" or task == "selective_alignment":
            file_name = f"{LOG_DIR}/results/{task}.csv"
        else:
            file_name = f"{LOG_DIR}/results/effectiveness.csv"

    elif task == "simple_prompt" or task == "diverse_prompt" or task == "MS-COCO":
        file_name = f"{LOG_DIR}/results/faithfulness.csv"

    else:
        raise ValueError(f"Invalid task: {task}")

    # log the result
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            if metric:
                f.write("method, target, seed, score, metric\n")
                f.write(f"{method}, {target}, {seed}, {score}, {metric}\n")
            else:
                f.write("method, target, seed, score\n")
                f.write(f"{method}, {target}, {seed}, {score}\n")

    else:
        with open(file_name, "a") as f:
            if metric:
                f.write(f"{method}, {target}, {seed}, {score}, {metric}\n")
            else:
                f.write(f"{method}, {target}, {seed}, {score}\n")
