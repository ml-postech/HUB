import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk import pos_tag
import torch
from PIL import Image
import clip
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
import argparse
import logging
import pandas as pd
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_noun(word, stop_words):
    if len(word) <= 2:  
        return False
    if word.lower() in stop_words:  
        return False
    tag = pos_tag([word])[0][1] 
    if tag in ['PRP', 'PRP$', 'WP', 'WP$']:  
        return False
    synsets = wn.synsets(word, pos='n')
    def is_descendant_of_physical_entity(synset):
        """
        Check if the given synset is a descendant of 'physical_entity.n.01'
        """
        physical_entity = wn.synset("physical_entity.n.01")  # 기준점
        return any(physical_entity in path for path in synset.hypernym_paths())
    if synsets[0].name().endswith(f'{word}.n.01') and is_descendant_of_physical_entity(synsets[0]):
        return True

    return False

def get_noun_dataset():
    """
    Get all nouns (including named entities) under physical_entity.n.01 in WordNet,
    without any additional filtering.
    """
    # Download necessary resources
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    stop_words = set(stopwords.words("english"))
    
    physical_entity = wn.synset("physical_entity.n.01")

    def closure_including_instances(synset):
        """Yield all hyponyms + instance hyponyms (transitively) for the given synset."""
        visited = set()
        stack = [synset]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                yield current
                # Normal hyponyms
                for h in current.hyponyms():
                    stack.append(h)
                for h in current.instance_hyponyms():
                    stack.append(h)

    # Collect all lemma names from the physical_entity subtree
    all_lemmas = []
    for s in closure_including_instances(physical_entity):
        if s.pos() == "n":
            for lemma in s.lemmas():
                lemma_name = lemma.name().lower()
                all_lemmas.append(lemma_name)

    # Remove duplicates by converting to a set
    unique_lemmas = list(set(all_lemmas))
    filtered_lemmas = [lemma for lemma in unique_lemmas if is_noun(lemma, stop_words)]
    logger.info(f"Found {len(filtered_lemmas)} unique lemmas under 'physical_entity.n.01'")

    return filtered_lemmas


def rank_nouns(nouns, target_concept):
    """Rank nouns by CLIP similarity to target concept"""
    
    # Load CLIP model
    logger.info("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    logger.info(f"Using device: {device}")
    
    # Prepare text templates
    target_text = f"a photo of {target_concept}"
    noun_texts = [f"a photo of {noun}" for noun in nouns]
    
    # Encode texts
    logger.info("Computing CLIP similarities...")
    with torch.no_grad():
        target_encoding = model.encode_text(clip.tokenize(target_text).to(device))
        similarities = []
        
        # Process nouns in batches
        batch_size = 256
        for i in tqdm(range(0, len(noun_texts), batch_size)):
            batch = clip.tokenize(noun_texts[i:i + batch_size]).to(device)
            batch_encodings = model.encode_text(batch)
            batch_similarities = torch.cosine_similarity(batch_encodings, target_encoding)
            similarities.extend(batch_similarities.cpu().numpy())
    
    # Create noun-similarity pairs and sort
    noun_similarities = list(zip(nouns, similarities))
    ranked_nouns = sorted(noun_similarities, key=lambda x: x[1], reverse=True)
    logger.info("Finished ranking nouns by similarity")
    
    return ranked_nouns[:1000]  # Return top 1000 nouns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    
    args = parser.parse_args()
    
    logger.info(f"Starting over-erasure detection for concept: {args.target}")

    nouns = get_noun_dataset()
    ranked_nouns = rank_nouns(nouns, args.target)
    # print(ranked_nouns)
    project_dir = os.getcwd()
    target_dir = os.path.join(project_dir, "prompts", "pinpoint_ness")
    os.makedirs(target_dir, exist_ok=True)
    pd.DataFrame(ranked_nouns, columns=['noun', 'similarity']).to_csv(f'{target_dir}/{args.target}.csv', index=False)
    
    logger.info("Pipeline completed successfully")