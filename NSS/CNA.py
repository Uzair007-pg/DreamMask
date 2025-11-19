"""
CNA.py — Category Name Association Module (COCO-based)

This module:
1. Loads all category names from the COCO instances annotation file.
2. Uses an LLM to generate novel categories that are semantically related.
3. Saves:
   - coco_categories.json      (C_train)
   - novel_categories.json     (C_novel)
"""

import json
from typing import List, Dict
from openai import OpenAI

# Default COCO annotation path (you can change this if needed)
COCO_ANN_PATH = "./data/coco/annotations/instances_train2017.json"

client = OpenAI()


def load_coco_categories(coco_ann_path: str = COCO_ANN_PATH) -> List[str]:
    """
    Load COCO categories from the instances annotation JSON.

    Args:
        coco_ann_path: Path to COCO instances_train2017.json

    Returns:
        Sorted list of unique category names (C_train).
    """
    with open(coco_ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    if "categories" not in ann:
        raise KeyError(
            f"'categories' field not found in {coco_ann_path}. "
            "Make sure this is a valid COCO instances annotation file."
        )

    names = {cat["name"] for cat in ann["categories"]}
    return sorted(names)


def generate_related_categories(
    base_categories: List[str],
    n_repeat: int = 5,
    top_k: int = 30
) -> List[str]:
    """
    Generate novel categories related to the base categories using an LLM.

    Args:
        base_categories: Known training categories (C_train)
        n_repeat: Number of LLM calls to stabilize the output
        top_k: Number of candidate categories per call

    Returns:
        Sorted list of stable, deduplicated novel categories (C_novel).
    """

    prompt = f"""
    You are an AI assistant.
    Given the following known object categories (from the COCO dataset):
    {', '.join(base_categories)}

    Generate {top_k} novel object categories that are semantically related and
    frequently co-occur with them in real-world scenes.

    Requirements:
    - Only output a comma-separated list of nouns.
    - Do not explain your reasoning.
    - Do not output anything other than the list.
    """

    collected: List[str] = []

    for _ in range(n_repeat):
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )
        text = response.output[0].content[0].text
        items = [x.strip().lower() for x in text.split(",") if x.strip()]
        collected.extend(items)

    # Count frequency to get stable categories
    freq: Dict[str, int] = {}
    for c in collected:
        freq[c] = freq.get(c, 0) + 1

    # Keep categories that appear at least twice and are not in the base set
    novel_categories = [
        k for k, v in freq.items()
        if v >= 2 and k not in base_categories
    ]

    return sorted(list(set(novel_categories)))


def save_categories(categories: List[str], output_path: str) -> None:
    """
    Save a category list to JSON.

    Args:
        categories: List of category names
        output_path: Path to output JSON file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)
    print(f"[INFO] Saved {len(categories)} categories to {output_path}")


if __name__ == "__main__":
    print(f"[INFO] Loading COCO categories from {COCO_ANN_PATH} ...")
    base = load_coco_categories(COCO_ANN_PATH)
    print(f"[INFO] Loaded {len(base)} base categories from COCO.")

    print("[INFO] Generating novel categories with LLM...")
    novel = generate_related_categories(base)
    print(f"[INFO] Generated {len(novel)} novel categories.")

    save_categories(base, "coco_categories.json")
    save_categories(novel, "novel_categories.json")

    print("[INFO] Done.")