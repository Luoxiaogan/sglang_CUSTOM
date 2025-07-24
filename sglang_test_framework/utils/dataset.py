"""Dataset utilities for SGLang testing framework.

Adapted from SGLang's dataset handling.
Source references:
- python/sglang/test_utils.py
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


DATASET_URLS = {
    "sharegpt": "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
    "alpaca": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
}


def download_dataset(dataset_name: str, cache_dir: str = None) -> str:
    """Download a dataset if not already cached.
    
    Args:
        dataset_name: Name of the dataset
        cache_dir: Directory to cache datasets
        
    Returns:
        Path to the downloaded dataset
    """
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "sglang" / "datasets"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename
    url = DATASET_URLS[dataset_name]
    filename = url.split("/")[-1]
    filepath = cache_dir / filename
    
    if filepath.exists():
        logger.info(f"Using cached dataset: {filepath}")
        return str(filepath)
    
    # Download the dataset
    logger.info(f"Downloading {dataset_name} dataset to {filepath}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dataset_name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    return str(filepath)


def load_dataset(
    dataset_path: str,
    dataset_name: str = "sharegpt",
    num_samples: int = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load dataset samples.
    
    Args:
        dataset_path: Path to the dataset file
        dataset_name: Name/format of the dataset
        num_samples: Number of samples to load (None for all)
        seed: Random seed for sampling
        
    Returns:
        List of dataset samples
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    
    if dataset_name == "sharegpt":
        # Extract conversations from ShareGPT format
        for item in data:
            conversations = item.get("conversations", [])
            if len(conversations) >= 2:
                # Use first user message and first assistant response
                user_msg = None
                assistant_msg = None
                
                for conv in conversations:
                    if conv.get("from") == "human" and user_msg is None:
                        user_msg = conv.get("value", "")
                    elif conv.get("from") == "gpt" and assistant_msg is None:
                        assistant_msg = conv.get("value", "")
                    
                    if user_msg and assistant_msg:
                        break
                
                if user_msg and assistant_msg:
                    samples.append({
                        "prompt": user_msg,
                        "completion": assistant_msg,
                        "id": item.get("id", f"sample_{len(samples)}")
                    })
    
    elif dataset_name == "alpaca":
        # Extract from Alpaca format
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            if input_text:
                prompt = f"{instruction}\\n\\nInput: {input_text}"
            else:
                prompt = instruction
            
            samples.append({
                "prompt": prompt,
                "completion": output,
                "id": f"alpaca_{len(samples)}"
            })
    
    # Sample if needed
    if num_samples and num_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, num_samples)
    
    logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
    return samples


def generate_synthetic_samples(
    num_samples: int,
    input_len_range: Tuple[int, int] = (100, 2000),
    output_len_range: Tuple[int, int] = (50, 500),
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Generate synthetic samples for testing.
    
    Args:
        num_samples: Number of samples to generate
        input_len_range: Range for input lengths (min, max)
        output_len_range: Range for output lengths (min, max)
        seed: Random seed
        
    Returns:
        List of synthetic samples
    """
    random.seed(seed)
    samples = []
    
    for i in range(num_samples):
        input_len = random.randint(*input_len_range)
        output_len = random.randint(*output_len_range)
        
        # Generate placeholder text
        prompt = f"Test prompt {i}: " + "x" * (input_len - 20)
        completion = f"Test response {i}: " + "y" * (output_len - 20)
        
        samples.append({
            "prompt": prompt,
            "completion": completion,
            "id": f"synthetic_{i}",
            "input_len": input_len,
            "output_len": output_len
        })
    
    return samples