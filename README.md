#  <img src="assets/icon.png" width="25px"> HUB: Holistic Unlearning Benchmark

This repository contains the original code for the [Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](https://arxiv.org/abs/2410.05664).


## News
[2024.10.29] We released HUB: Holistic Unlearning Benchmark :fire:

---
## Environment setup
### Installation
To set up the environment, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/ml-postech/HUB.git
    cd HUB
    ```
2.	Create and activate the conda environment:
    ```bash
    conda env create -f environment.yaml
    conda activate HUB
    ```

### Update `envs.py`
Before running the code, make sure to update envs.py with the correct file paths, GPT API configurations, and any other parameters specific to your environment.

### Download pre-trained models and datasets
- [Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
  - For aesthetic score, we use the `sac+logos+ava1-l14-linearMSE.pth` model. Place it in the `/models/aesthetic_predictor` directory.

- [Datasets for application tasks](https://drive.google.com/file/d/1JMUeDOfZGtygOmqxjTPqcVkfyWrq8aII/view?usp=sharing)
    - Extract the contents of the zip file and place them in the `/datasets`.
- [MNIST LeNet5](https://drive.google.com/file/d/1UH9aHKQdnNCzey_WZKDh47P4_5KLZr0o/view?usp=sharing)
    - Place the file in the `/models/mnist`.
- [MNIST pretrained model](https://drive.google.com/file/d/1BFf_MIt2P6KOGtHgGjvDnAxS-grNRGFJ/view?usp=sharing)
    - You can use this pretrained model for your method.

---
## Evaluation

### Run the evaluation
To run the evaluation, execute the following command:
```bash
python main.py --config YOUR_CONFIG.yaml
```
### Batch to log
We use GPT-4 to evaluate whether the generated images contain specific concepts. For more efficient evaluation, we use [batch API](https://platform.openai.com/docs/guides/batch). After sending the queries and completing the evaluations, run the following command to organize the logs and results. Replace `NUM_BATCHES` with the number of batches you want to evaluate.

```bash
python batch2log.py --num_batches NUM_BATCHES
```
### How to evaluate own model?
For now, we support the following six unlearning methods: [AC](https://arxiv.org/abs/2303.13516), [SA](https://arxiv.org/abs/2305.10120), [SalUn](https://arxiv.org/abs/2310.12508), [UCE](https://arxiv.org/abs/2308.14761), [ESD](https://arxiv.org/abs/2303.07345), [Receler](https://arxiv.org/abs/2311.17717). To evaluate your own model, you need to modify `model.__init__.py` to include the loading of your custom model. We recommend that you place your model in `models/sd/YOUR_METHOD/`

---
## How to evaluate each task individually?
To evaluate each task separately, follow these commands. In the following examples, replace the variables according to the settings you want to evaluate. Make sure to execute below command before evaluating each task.
```bash
export PYTHONPATH=$PYTHONPATH:PROJECT_DIR
```

### Effectiveness, Faithfulness, Compliance, and Over-erasing effect
For effectiveness, faithfulness and compliacnce task, we have to generate images first with below command. `TASK` should be one of `simple_prompt`, `diverse_prompt`, `MS-COCO`, `selective_alignment` or `over_erasing`.
```bash
python source/image_generation.py --task TASK --method METHOD --target TARGET 
```

After generate images, execute below command
```bash
python source/utils/evaluation_batch.py --task TASK --method METHOD --target TARGET --seed SEED
```

### Side effects: Model bias
```bash
python source/bias/bias.py \
    --method METHOD \
    --target TARGET \
    --batch_size BATCH_SIZE \
    --device DEVICE \
    --seed SEED
```

### Downstream application
**Sketch-to-image**
```bash
python source/image_translation/image_translation.py \
    --method METHOD \
    --target TARGET \
    --task sketch2image \
    --device DEVICE \
    --seed SEED
```

**Image-to-image**
```bash
python source/image_translation/image_translation.py \
    --method METHOD \
    --target TARGET \
    --task image2image \
    --device DEVICE \
    --seed SEED
```

**Concept restoration**
```bash
python source/concept_restoration/concept_restoration.py \
    --method METHOD \
    --target TARGET \
    --start_t_idx START_T_IDX \ # check the description in the code
    --device DEVICE \
    --seed SEED
```

After generate images, execute below command. `TASK` should be one of `sketch2image`, `image2image`, or `concept_restoration`.
```bash
python source/utils/evaluation_batch.py --task TASK --method METHOD --target TARGET --seed SEED
```

---
## Citation
    @article{moon2024holistic,
        title={Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning},
        author={Moon, Saemi and Lee, Minjong and Park, Sangdon and Kim, Dongwoo},
        journal={arXiv preprint arXiv:2410.05664},
        year={2024}
    }

