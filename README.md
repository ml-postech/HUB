#  [ICCV 2025] HUB: Holistic Unlearning Benchmark

Official Implementation of [Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](https://arxiv.org/abs/2410.05664) (ICCV 2025)
- A comprehensive benchmark for evaluating unlearning methods in text-to-image diffusion models across multiple tasks and metrics
- 💡 Feel free to explore the code, open issues, or reach out for discussions and collaborations!

## 📦 Environment setup
### Installation
To set up the environment, follow these steps:
1. Clone the repository:
    ```
    git clone https://github.com/ml-postech/HUB.git
    cd HUB
    ```
2.	Create and activate the conda environment:
    ```bash
    conda create -n HUB python=3.9
    conda activate HUB
    pip install -r requirements.txt
    ```

### Download pre-trained models and datasets
- [Reference image dataset](https://huggingface.co/datasets/hi-sammy/HUB_reference_images)
    - To evaluate target proportion, reference images for each concept are required. We provide these reference images as part of a Hugging Face dataset.
    - Once downloaded, place the dataset under the `images/` directory:

- [Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
  - For aesthetic score, we use the `sac+logos+ava1-l14-linearMSE.pth` model. 
  - Place it in the `/models/aesthetic_predictor` directory.

- [Q16](https://github.com/ml-research/Q16?tab=readme-ov-file)
    - Download `prompts.p` from [this link](https://drive.google.com/file/d/1lWKdUTvPDWY9hw7ruDdCHXMOqs24PbQq/view?usp=sharing) and place it at `/models/q16/` directory.

- [GIPHY Celebrity Detector](https://github.com/Giphy/celeb-detection-oss)
    - Download giphy_celeb_detector.zip from [this link](https://drive.google.com/file/d/1e1S4hDsqHkMBkSBSLAcuyLtpxhVFlbGg/view?usp=sharing) and extract it to `/models/` directory.



## 🖼️ Image generation
To perform evaluation using HUB, you must first generate images for each concept and task with your unlearned model. Use the prompts described below to generate images.

```
python source/image_generation.py \
    --method YOUR_METHOD \
    --target TARGET \
    --task TASK
```

`TASK` must be one of the following: `target_image`, `general_image`, `selective_alignment`, `pinpoint_ness`, `multilingual_robustness`, `attack_robustness`, `incontext_ref_image`.


## 💬 Prompt generation
All prompts used in our experiments are provided in the `prompts/` directory.
You can also generate prompts for your own target using the following scripts.


### Target prompt generation (base prompts)
```
python source/prompt_generation/prompt.py \
  --target YOUR_TARGET \
  [--style] [--nsfw]
```
- Use `--style` for style-related targets
- Use `--nsfw` for NSFW-related targets

### Multilingual robustness
After generating the base prompts, create multilingual versions:
```
python source/prompt_generation/translate_prompt.py \
  --target YOUR_TARGET
```

### Pinpoint-ness
```
python source/prompt_generation/pinpoint_ness.py \
  --target YOUR_TARGET
  ```

### Selective alignment
```
python source/prompt_generation/selective_alignment.py \
  --target YOUR_TARGET \
  [--style]   # Add only if this is a style-related target
```

## 📊 Evaluation
### How to evaluate own model?
For now, we support the following seven unlearning methods: [SLD](https://arxiv.org/abs/2211.05105), [AC](https://arxiv.org/abs/2303.13516), [ESD](https://arxiv.org/abs/2303.07345), [UCE](https://arxiv.org/abs/2308.14761), [SA](https://arxiv.org/abs/2305.10120), [Receler](https://arxiv.org/abs/2311.17717), [MACE](https://arxiv.org/abs/2403.06135). To evaluate your own model, you need to modify `model.__init__.py` to include the loading of your custom model. We recommend that you place your model in `models/sd/YOUR_METHOD/`.

### Run the evaluation
To run the all tasks at once, execute the following command:
```bash
python main.py --method YOUR_METHOD --target TARGET
```

## 🎯 How to evaluate each task individually?
Running evaluation using `main.py` takes a long time, as it evaluates all tasks at once. To evaluate each task separately, follow these commands. In the following examples, replace the variables according to the settings you want to evaluate. Make sure to execute below command before evaluating each task.
```bash
export PYTHONPATH=$PYTHONPATH:YOUR_PROJECT_DIR
```

### Target proportion, multilingual robustness, and attack robustness
The evaluation code is configured to run separately for each concept type, because different classifiers are used. For the `target_proportion`, `multilingual_robustness`, and `attack_robustness` tasks, run the following code.

* **Celebrity**
```
python source/eval/eval_gcd.py \
    --task TASK \
    --method YOUR_METHOD \
    --target TARGET
```

* **Style, IP (VLM)**

```
python source/eval/eval_vlm.py \
    --task TASK \
    --method YOUR_METHOD \
    --target TARGET
```

* **NSFW**

```
python source/eval/eval_nsfw.py \
    --task TASK \
    --method YOUR_METHOD \
    --target TARGET
```

### Quality & Alignment
- TASK: `general_image`, `target_image`.
- METRIC: `aesthetic`, `ImageReward`, `PickScore`, `FID`, `FID_SD`.

```
python source/quality/evaluation.py \
    --method YOUR_METHOD \
    --target TARGET \
    --task TASK \
    --metric METRIC
```

### Selective alignment

```
python source/eval/eval_selective_alignment.py \
    --method YOUR_METHOD \
    --target TARGET
```

### Pinpoint-ness

```
python source/eval/eval_vlm.py \
    --task "pinpoint_ness" \
    --method YOUR_METHOD \
    --target TARGET
```

## 📌 To Do
- [ ] Add two attacks.
- [ ] Add a leaderboard for each task.
- [ ] Add new unlearning methods.


## 📚 Citation
    @article{moon2024holistic,
        title={Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning},
        author={Moon, Saemi and Lee, Minjong and Park, Sangdon and Kim, Dongwoo},
        journal={arXiv preprint arXiv:2410.05664},
        year={2024}
    }

