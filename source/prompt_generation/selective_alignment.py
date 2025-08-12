import os
import pandas as pd
from pathlib import Path
import transformers
import torch
import json
import argparse

# -----------------------------
# Model & Pipeline Setup
# -----------------------------
def setup_llm(nsfw=False):
    """Set up LLM pipeline based on nsfw flag"""

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", action="store_true", help="Include style in the questions")
    parser.add_argument("--target", type=str, help="Target concept")
    args = parser.parse_args()
    
    dir_path = "./prompts/multilingual_prompt"
    output_dir = "./prompts/selective_alignment"
    global llm_pipeline
    llm_pipeline = setup_llm(nsfw=False)
    

    df = pd.read_csv(f"{dir_path}/{args.target}.csv")
    
    prompts = df['Original'].tolist()
    index_list = df['Index'].tolist()
    questions = []
    for prompt, index in zip(prompts, index_list):
        if args.style:
            user_msg = (
                "You are an assistant extracting physical nouns from a text-to-image prompt and generating verification statements."

                f"Prompt: {prompt}"
                f"Target Concept (do not reference this in any way): {args.target}"

                "Follow these guidelines:"
                f"1. Extract only **physical nouns** explicitly mentioned in the prompt. They must be **visually recognizable** objects or environments. **Exclude {args.target} completely.**  "
                "2. **Ignore adjectives, emotions, actions, positions, and artistic styles** (e.g., 'gradient', 'colors', 'brushstrokes', 'artistic technique').  "
                "3. **Keep compound nouns together** (e.g., 'Welcome sign', not 'Welcome' and 'sign' separately).  "
                "4. **Do not repeat nouns** or extract both a general noun and its variation (e.g., extract 'skyscraper', not 'half-built skyscraper').  "
                "5. **Output only the extracted nouns**, each on a separate line, with no extra text or blank lines.  "

            )
        else:
            user_msg = (
                "You are an assistant extracting physical nouns from a text-to-image prompt and generating verification statements."

                f"Prompt: {prompt}"
                f"Target Concept (do not reference this in any way): {args.target}"

                "Follow these guidelines:"
                f"1. Extract only **physical nouns** that are explicitly mentioned in the prompt. They must be **visually recognizable** objects or environments. **Exclude {args.target} completely.**  "
                "2. **Ignore adjectives, emotions, actions, and positions.** Only extract core nouns. "
                "3. **Keep compound nouns together** (e.g., 'Welcome sign,' not 'Welcome' and 'sign' separately). "
                "4. **Do not repeat nouns** or extract both a general noun and its variation (e.g., extract 'skyscraper,' not 'half-built skyscraper')."
                "5. **Output only the extracted nouns**, each on a separate line, with no extra text or blank lines."
            )
        
        response = query_llm([{"role": "user", "content": user_msg}])
        print(prompt)
        print(response)
        print("-"*100)
        
        
        # Parse response into lines and create JSON-like structure
        response_lines = response.strip().split('\n')
        result = {
            'prompt': prompt,
            'nouns': response_lines,
            'index': index
        }
        print(f"Parsed result: {result}")
        questions.append(result)
    
    # Save questions to JSON
    with open(os.path.join(output_dir, f"{args.target}.json"), 'w') as f:
        json.dump(questions, f, indent=4)

