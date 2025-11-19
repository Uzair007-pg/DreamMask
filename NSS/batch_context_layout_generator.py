"""
batch_context_layout_generator.py
---------------------------------
Batch context-aware layout generation using GPT-5 and rendering to layout images.

Pipeline:
1. Load base categories from COCO (coco_categories.json).
2. Load novel categories from CNA (novel_categories.json).
3. Build a combined vocabulary: C_all = C_train ∪ C_novel.
4. For each scene:
    - Sample a subset of categories from C_all.
    - Ask GPT-5 to create a realistic, context-aware scene layout in HTML
      with absolutely-positioned boxes.
    - Parse the HTML to extract (x, y, w, h, name).
    - Render the layout into a PNG image.
    - Save HTML + PNG + JSON metadata.

Dependencies:
    pip install openai beautifulsoup4 pillow

Environment:
    export OPENAI_API_KEY="your_api_key_here"
"""

import os
import json
import random
import re
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

client = OpenAI()


# ============================================================
# 1. Load categories (COCO + novel)
# ============================================================

def load_category_list(path: str) -> List[str]:
    """
    Load a category list from JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vocabulary(
    coco_categories_path: str = "coco_categories.json",
    novel_categories_path: str = "novel_categories.json"
) -> Dict[str, List[str]]:
    """
    Build vocabulary from COCO base categories and novel categories.

    Returns:
        {
          "coco": [ ... ],       # C_train
          "novel": [ ... ],      # C_novel
          "all": [ ... ]         # C_train ∪ C_novel
        }
    """
    coco_categories = load_category_list(coco_categories_path)
    novel_categories = load_category_list(novel_categories_path)

    all_categories = sorted(list(set(coco_categories) | set(novel_categories)))

    print(f"[INFO] Loaded {len(coco_categories)} COCO base categories.")
    print(f"[INFO] Loaded {len(novel_categories)} novel categories.")
    print(f"[INFO] Combined vocabulary size: {len(all_categories)}")

    return {
        "coco": coco_categories,
        "novel": novel_categories,
        "all": all_categories
    }


# ============================================================
# 2. Sampling scene categories
# ============================================================

def select_scene_categories(
    categories: List[str],
    min_objects: int = 5,
    max_objects: int = 8,
    seed: Optional[int] = None
) -> List[str]:
    """
    Randomly select a subset of categories to appear in a single scene.
    """
    if seed is not None:
        random.seed(seed)

    if not categories:
        return []

    n = random.randint(min_objects, max_objects)
    n = min(n, len(categories))
    if len(categories) <= n:
        return categories
    return random.sample(categories, n)


# ============================================================
# 3. GPT-5 HTML layout generation (context-aware)
# ============================================================

def generate_layout_html_with_context(
    scene_categories: List[str],
    all_categories: List[str],
    width: int = 1024,
    height: int = 1024
) -> str:
    """
    Generate context-aware HTML scene layout using GPT-5.

    The HTML must contain <div> blocks like:
        <div class="object"
             data-name="plate"
             style="position:absolute; left:100px; top:200px; width:160px; height:100px;">
        </div>
    """
    all_preview = ", ".join(all_categories[:40])
    scene_preview = ", ".join(scene_categories)

    prompt = f"""
You are a scene layout planner for 2D images.

Image canvas size: {width}x{height} pixels.

You are given a set of object categories that must appear in the scene:
SCENE_CATEGORIES = [{scene_preview}]

You also know the broader vocabulary of possible objects:
ALL_CATEGORIES (subset shown) = [{all_preview}, ...]

Your task:
1. Imagine a realistic scene (indoor or outdoor) that uses ALL of SCENE_CATEGORIES.
2. Place each object in a spatially plausible way, using your world knowledge
   about how objects typically relate (for example:
   - plate on a table
   - chair around a table
   - car on a road
   - tree on grass
   - lamp near a sofa
   - pillow on a bed
   etc.).
3. Avoid impossible placements such as:
   - sofa floating in the sky
   - car inside a small mug
   - random massive overlaps that make objects indistinguishable.

Output format:
- You MUST output ONLY raw HTML, nothing else.
- Use a root container: <div id="scene"> ... </div>
- For each object, output a <div> with:
    class="object"
    data-name="<category_name>"
    style="position:absolute; left:Xpx; top:Ypx; width:Wpx; height:Hpx;"
- Use integer pixel values for X, Y, W, H.
- Ensure every category in SCENE_CATEGORIES appears exactly once as a <div class="object">.
- Make sure boxes are within the {width}x{height} canvas and do not completely overlap.

Example of a single object (for format only, DO NOT reuse the numbers):
<div class="object" data-name="plate"
     style="position:absolute; left:120px; top:300px; width:180px; height:90px;"></div>
"""

    response = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )

    html = response.output[0].content[0].text
    return html


# ============================================================
# 4. HTML parsing
# ============================================================

def parse_layout_html(html: str) -> List[Dict[str, Any]]:
    """
    Parse GPT-generated HTML and extract bounding boxes.

    Each object is represented as:
        {
            "name": <category name>,
            "x": int,
            "y": int,
            "w": int,
            "h": int
        }
    """
    soup = BeautifulSoup(html, "html.parser")
    boxes: List[Dict[str, Any]] = []

    for div in soup.find_all("div", class_="object"):
        name = div.get("data-name", "unknown")
        style = div.get("style", "")

        def extract(prop: str) -> int:
            # e.g., "left:120px"
            match = re.search(rf"{prop}\s*:\s*(\d+)px", style)
            return int(match.group(1)) if match else 0

        x = extract("left")
        y = extract("top")
        w = extract("width")
        h = extract("height")

        if w <= 0 or h <= 0:
            continue

        boxes.append({"name": name, "x": x, "y": y, "w": w, "h": h})

    return boxes


# ============================================================
# 5. Layout rendering
# ============================================================

def color_from_name(name: str) -> tuple:
    """
    Generate a pseudo-random but stable color from the object name.
    """
    random.seed(hash(name) & 0xFFFFFFFF)
    r = 50 + random.randint(0, 205)
    g = 50 + random.randint(0, 205)
    b = 50 + random.randint(0, 205)
    return (r, g, b)


def render_layout_image(
    boxes: List[Dict[str, Any]],
    width: int = 1024,
    height: int = 1024,
    save_path: str = "layout.png"
) -> None:
    """
    Render bounding boxes with labels into a PNG image.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        name = box["name"]
        color = color_from_name(name)

        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        label = name
        text_w, text_h = draw.textsize(label, font=font)
        bg_x2 = min(x + text_w + 6, width)
        bg_y2 = min(y + text_h + 6, height)
        draw.rectangle([x, y, bg_x2, bg_y2], fill=(255, 255, 255))
        draw.text((x + 3, y + 3), label, fill=color, font=font)

    img.save(save_path)
    print(f"[INFO] Layout image saved to {save_path}")


# ============================================================
# 6. Batch generation
# ============================================================

def generate_batch_layouts(
    coco_categories_path: str = "coco_categories.json",
    novel_categories_path: str = "novel_categories.json",
    out_dir: str = "layouts_dataset",
    num_scenes: int = 100,
    canvas_width: int = 1024,
    canvas_height: int = 1024,
    min_objects: int = 5,
    max_objects: int = 8,
) -> None:
    """
    Batch-generate many context-aware layouts.

    For each scene i:
        - Pick scene categories from C_all.
        - Call GPT-5 to generate HTML.
        - Parse boxes.
        - Render PNG.
        - Save:
            - HTML:  scene_{i:05d}.html
            - PNG:   scene_{i:05d}.png
            - META:  scene_{i:05d}.json
    """
    os.makedirs(out_dir, exist_ok=True)

    vocab = build_vocabulary(coco_categories_path, novel_categories_path)
    all_categories = vocab["all"]

    for i in range(num_scenes):
        print(f"\n[SCENE {i+1}/{num_scenes}] ----------------------------------")

        scene_categories = select_scene_categories(
            all_categories,
            min_objects=min_objects,
            max_objects=max_objects
        )
        if not scene_categories:
            print("[WARN] No scene categories selected, skipping.")
            continue

        print("[INFO] Scene categories:", scene_categories)

        # Generate HTML via GPT-5
        try:
            html = generate_layout_html_with_context(
                scene_categories=scene_categories,
                all_categories=all_categories,
                width=canvas_width,
                height=canvas_height
            )
        except Exception as e:
            print("[ERROR] Failed to call GPT-5:", e)
            continue

        html_path = os.path.join(out_dir, f"scene_{i:05d}.html")
        png_path = os.path.join(out_dir, f"scene_{i:05d}.png")
        meta_path = os.path.join(out_dir, f"scene_{i:05d}.json")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[INFO] HTML saved to {html_path}")

        boxes = parse_layout_html(html)
        print(f"[INFO] Parsed {len(boxes)} objects from HTML.")

        if not boxes:
            print("[WARN] No valid boxes parsed, skipping PNG/meta.")
            continue

        try:
            render_layout_image(
                boxes,
                width=canvas_width,
                height=canvas_height,
                save_path=png_path
            )
        except Exception as e:
            print("[ERROR] Failed to render layout:", e)
            continue

        meta = {
            "scene_index": i,
            "scene_categories": scene_categories,
            "boxes": boxes,
            "html_path": os.path.abspath(html_path),
            "png_path": os.path.abspath(png_path),
            "canvas_width": canvas_width,
            "canvas_height": canvas_height
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Metadata saved to {meta_path}")


# ============================================================
# 7. CLI entry
# ============================================================

if __name__ == "__main__":
    generate_batch_layouts(
        coco_categories_path="coco_categories.json",
        novel_categories_path="novel_categories.json",
        out_dir="layouts_dataset",
        num_scenes=50,          # change to 500/1000 if you like
        canvas_width=1024,
        canvas_height=1024,
        min_objects=5,
        max_objects=8,
    )
    print("\n[INFO] Batch generation finished.")