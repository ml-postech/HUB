import json
import os
import re
import argparse
from tqdm import tqdm
from openai import OpenAI

from envs import API_KEY, IMG_DIR
from source.utils import setup_logging, log_result, set_logger


# num_batches: batch number for logging results from recent batches
def batch2log(num_batches):
    client = OpenAI(api_key=API_KEY)
    logger = set_logger()

    batch_list = client.batches.list(limit=num_batches)

    # * See https://platform.openai.com/docs/api-reference/batch
    for b in tqdm(batch_list.data):
        batch_id = b.id
        batch_description = b.metadata["description"]
        batch_description = os.path.splitext(os.path.basename(batch_description))[0]
        task, method, target, seed = batch_description.split("-")

        try:
            output_file_id = b.output_file_id
            output_file = client.files.content(output_file_id).text
        except:
            logger.info(f"Error: {batch_id}, {batch_description}")
            continue

        logger.info(f"Processing batch: {batch_id}, {batch_description}")
        input_dir = os.path.join(IMG_DIR, task, method, target, seed)

        # set up logging
        log_file = setup_logging(task, method, target, seed)
        logger.info(f"Log file: {log_file}")

        with open(log_file, "w") as f:
            acc_result = []
            # * See https://community.openai.com/t/error-retrieving-content-of-a-embedding-batch-job/854900/2
            for line in output_file.split("\n")[:-1]:
                data = json.loads(line)
                custom_id = data.get("custom_id")
                answer = data["response"]["body"]["choices"][0]["message"]["content"]

                if re.search(r"\bno\b", answer, re.IGNORECASE):
                    acc = 0
                elif re.search(r"\byes\b", answer, re.IGNORECASE):
                    acc = 1
                else:
                    acc = 0

                acc_result.append(acc)

                f.write(f"{custom_id},{answer}\n")

            acc_result = sum(acc_result) / len(acc_result)

            logger.info(f"Accuracy: {acc_result}")
            log_result(acc_result, task, method, target, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_batches", type=int, default=1, help="number of batches to process"
    )
    args = parser.parse_args()
    batch2log(args.num_batches)
