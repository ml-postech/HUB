import argparse
import csv
import torch
import transformers
import random
import os


def setup_llm(model_id):
    """Set up LLM pipeline with given model ID"""
    print(f"Setting up LLM pipeline with model: {model_id}")
    llm_pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return llm_pipeline


def query_llm(pipeline, messages):
    """Query the LLM and return generated text"""
    output = pipeline(messages, max_new_tokens=4096)
    return output[0]["generated_text"][-1]["content"]


def translate_prompt(pipeline, prompt, target_lang):
    """Translate a single prompt to target language"""
    messages = [
        {
            "role": "user",
            "content": f"You are a professional translator. Translate the given text to {target_lang} naturally."
            f"Translate this prompt to {target_lang}:\n\n{prompt}"
            "Return only the translated text, no other text.",
        }
    ]
    return query_llm(pipeline, messages)


def main():
    parser = argparse.ArgumentParser(description="Translate prompts to target language")
    parser.add_argument(
        "--target", type=str, required=True, help="Target concept"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of prompts to sample and translate",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="LLM model ID to use",
    )
    args = parser.parse_args()

    print(f"Starting translation process with args: {args}")

    # Setup LLM
    llm_pipeline = setup_llm(args.model_id)

    # Read prompts from input CSV
    prompts = []
    with open(f"./prompts/target_image/{args.target}.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["prompt"])


    sampled_prompts = random.sample(prompts, min(args.num_samples, len(prompts)))
    print(f"Sampled {len(sampled_prompts)} prompts for translation")

    # Define all target languages
    languages = ["Spanish", "French", "German", "Italian", "Portuguese"]

    # Translate prompts to all languages
    all_translations = []
    # Create a dictionary to store translations for each prompt
    translations = {prompt: {} for prompt in sampled_prompts}

    # Translate each prompt to each language
    for language in languages:
        print(f"\nTranslating to {language}...")
        for i, prompt in enumerate(sampled_prompts, 1):
            translated = translate_prompt(llm_pipeline, prompt, language)
            print(
                f"Translated prompt {i}/{len(sampled_prompts)} to {language}: {translated}"
            )
            translations[prompt][language] = translated
        print(f"Completed translations for {language}")

    # Save translations to CSV with languages as columns
    print(f"Saving translations to {args.target}.csv")
    fieldnames = ["Original"] + languages
    with open(f"./prompts/multilingual_robustness/{args.target}.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prompt in sampled_prompts:
            row = {"Original": prompt}
            row.update(translations[prompt])
            writer.writerow(row)

    print(f"All translations saved to {args.output_csv}")


if __name__ == "__main__":
    main()
