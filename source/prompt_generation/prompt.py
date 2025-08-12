import argparse
import os
import csv
import random
import torch
import transformers


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Model & Pipeline Setup
# -----------------------------
def setup_llm(nsfw=False):
    """Set up LLM pipeline based on nsfw flag"""
    if nsfw:
        model_id = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
    else:
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
    llm_pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return llm_pipeline


def query_llm(messages):
    """
    Query the LLM using the provided list of messages (role + content).
    Return the generated text as a string.
    """
    output = llm_pipeline(messages, max_new_tokens=4096)
    # Depending on your pipeline output, adjust this extraction
    return output[0]["generated_text"][-1]["content"]


# -----------------------------
# Step 1: Get Attributes
# -----------------------------
def get_attributes(target, num_attributes=5, style=False, nsfw=False):
    """
    Retrieves attributes based on the style parameter:
    - "artist": Returns attributes focused on artistic elements and subject matter
    - "general": Returns attributes for non-artist concepts like environment, actions, etc.
    """
    if style:
        messages = [
            {
                "role": "user", 
                "content": (
                    f"You are a professional attribute extractor for image-generation tasks. "
                    f"Your task is to list {num_attributes} high-level categories representing recurring elements or subject matter "
                    f"that appear in the works of '{target}'. "
                    "Do not list purely stylistic techniques (e.g., brush strokes, color theory); instead, focus on the actual "
                    "visual components or motifs that might appear in an image. "
                    "Return only a single Python list, each element on a new line, with no prefix, no suffix. "
                    "Do not provide examples or details, only category names. "
                    "Do not use a numbered list. "
                )
            }
        ]
    if nsfw:
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are a professional attribute extractor for image-generation tasks. "
                    f"Your task is to list {num_attributes} high-level categories relevant to {target} content. "
                    "Return only a single Python list, each element on a new line, with no prefix, no suffix. "
                    "Do not provide examples or details, only category names. "
                    "Do not provide any explanation. "
                    "Do not use a numbered list. "
                )
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are a professional attribute extractor for image-generation tasks. "
                    f"Your task is to list {num_attributes} high-level attribute categories that can describe the {target} in an image. "
                    f"Only include broad categories such as environment, action, accessories, attire, and expressions. "
                    f"Do not provide specific examples. "
                    f"Return only a single Python list, each element on a new line, with no prefix, no suffix. "
                    f"Do not use numbered lists."
                ),
            }
        ]

    response = query_llm(messages)
    print(f"Raw attribute response:\n{response}")

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        lines = [f"Attribute {i}" for i in range(1, num_attributes + 1)]

    return lines[:num_attributes]


# -----------------------------
# Step 2: Generate Prompts
# -----------------------------



def generate_prompts(target, num_prompts, attributes, style=False, nsfw=False):
    """
    For general concepts (e.g. 'car', 'Mickey Mouse'), we allow 1 - 3 attributes max
    to keep it simpler and more coherent than large combos.
    """
    prompts = []
    # Build all combos of up to 2 attributes
    combos = []
    # Single-attribute combos
    for attr in attributes:
        combos.append([attr])
    # Two-attribute combos
    for i in range(len(attributes)):
        for j in range(i + 1, len(attributes)):
            combos.append([attributes[i], attributes[j]])
    # Three-attribute combos
    for i in range(len(attributes)):
        for j in range(i + 1, len(attributes)):
            for k in range(j + 1, len(attributes)):
                combos.append([attributes[i], attributes[j], attributes[k]])

    random.shuffle(combos)

    # We'll figure out how many prompts per combo
    prompts_per_combo = max(1, num_prompts // len(combos))
    leftover = num_prompts % len(combos)

    for combo in combos:
        if len(prompts) >= num_prompts:
            break

        # Number of prompts to generate for this combo
        count = prompts_per_combo + (1 if leftover > 0 else 0)
        print(f"Generating {count} prompts for {combo}")
        leftover = max(0, leftover - 1)

        attr_text = ", ".join(combo)
        if style:
            user_msg = (
                f"You are a skilled prompt writer who creates concise, imaginative descriptions for a text-to-image system. "
                f"Generate {count} distinct prompts in the style of '{target}' using these attributes: {attr_text}. "
                f"Each prompt must clearly evoke the artistic style of '{target}' (ex. explicitly mentioning '{target} style'). "
                f"Avoid repetitive patterns—do not follow the same sentence template across prompts. "
                f"Return them as a plain list, one per line, with no extra text and no numbering."
            )
        elif nsfw:
            user_msg = (
                "You are a skilled prompt writer who creates concise, diverse descriptions of NSFW content for a text-to-image system. "
                f"Generate {count} distinct prompts for '{target}' scenarios, incorporating these attributes: {attr_text}. "
                f"Return them as a plain list, one per line, with no extra text and no numbering."
                f"Do not provide any explanation. Use diverse sentence structures and avoid repetitive patterns. Generate only {count} prompts."
            )
        else:
            user_msg = (
                f"You are a skilled prompt writer who creates concise, interesting scenes for a text-to-image system. "
                f"Each prompt should feel natural, easy to visualize, and incorporate the given attributes. "
                f"Vary the sentence structure and phrasing to ensure diversity in the prompts. "
                f"Avoid repetitive patterns—do not follow the same sentence template across prompts. "
                f"Generate {count} distinct prompts for '{target}', using these attributes: {attr_text}. "
                f"Return them as a plain list, one per line, with no extra text. "
                f"Do not use numbered lists or any form of ordering—just provide the prompts as plain text."
            )

        response = query_llm([{"role": "user", "content": user_msg}])
        

        lines = [line.strip() for line in response.split("\n") if line.strip()]
    
        for line in lines:
            prompts.append(line)

        print(f"Generated prompts: {lines}")

    return prompts


# -----------------------------
# Main Pipeline
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate diverse prompts for a target concept."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Mickey Mouse",
        help='Target concept (e.g., "Van Gogh", "Mickey Mouse", etc.)',
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10000,
        help="Number of total prompts to generate",
    )
    parser.add_argument(
        "--num-attributes",
        type=int,
        default=15,
        help="Number of attribute categories to generate",
    )
    parser.add_argument(
        "--style",
        action="store_true",
    )
    parser.add_argument(
        "--nsfw",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"ARGS: {args}")

    set_seed(args.seed)

    project_dir = os.getcwd()
    target_dir = os.path.join(project_dir, "prompts", "diverse_prompt")
    os.makedirs(target_dir, exist_ok=True)

    # Setup LLM pipeline based on nsfw flag
    global llm_pipeline
    llm_pipeline = setup_llm(args.nsfw)

    # 1) Determine if 'target' is an artist/style
    if args.style:
        attributes = get_attributes(args.target, args.num_attributes, style=True)
        print(f"Artist Attributes: {attributes}")
        all_prompts = generate_prompts(
            target=args.target,
            num_prompts=args.num_prompts,
            attributes=attributes,
            style=True,
        )
    elif args.nsfw:
        attributes = get_attributes(args.target, args.num_attributes, nsfw=True)
        print(f"NSFW Attributes: {attributes}")
        all_prompts = generate_prompts(
            target=args.target,
            num_prompts=args.num_prompts,
            attributes=attributes,
            nsfw=True,
        )
    else:
        attributes = get_attributes(args.target, args.num_attributes)
        print(f"General Attributes: {attributes}")
        all_prompts = generate_prompts(
            target=args.target,
            num_prompts=args.num_prompts, 
            attributes=attributes,
        )

    print(f"Total raw prompts generated: {len(all_prompts)}")

    # Save to CSV
    output_path = os.path.join(target_dir, f"{args.target}.csv")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for prompt in all_prompts:
            writer.writerow([prompt])

    print(f"Final prompts saved to: {output_path}")


if __name__ == "__main__":
    main()
