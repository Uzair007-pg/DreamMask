# DreamMask-style NSS (COCO-based) — CNA + Batch Layout Generation

This mini-project implements the **data-centric part** of DreamMask’s Novel Sample Synthesis (NSS) pipeline, based on the COCO dataset.

You will place the following two Python files in the **same folder**:

- `CNA.py`
- `batch_context_layout_generator.py`

The pipeline includes:

1. **`CNA.py` — Category Name Association (CNA)**  
   - Reads base categories (C_train) from the **COCO** instances annotation file  
   - Uses an LLM to generate semantically related **novel categories (C_novel)**  
   - Saves:
     - `coco_categories.json`
     - `novel_categories.json`

2. **`batch_context_layout_generator.py` — Batch Context-aware Layout Generation**  
   - Builds vocabulary `C_all = C_train ∪ C_novel`  
   - For each scene:
     - Samples a subset of categories from `C_all`
     - Asks GPT-5 to generate a **realistic, context-aware HTML layout**
     - Parses bounding boxes from the HTML
     - Renders a layout image (PNG)
     - Saves HTML + PNG + JSON metadata per scene

This corresponds to the **“category name association” + “context-aware layout planning”** part of DreamMask’s NSS pipeline.

---

## 1. Directory Structure & COCO Path

Assume the following project structure:

```text
project_root/
│
├── CNA.py
├── batch_context_layout_generator.py
├── README.md
│
└── data/
    └── coco/
        ├── images/
        │   ├── train2017/              # standard COCO structure (not required yet)
        │   └── val2017/
        └── annotations/
            ├── instances_train2017.json   # REQUIRED for CNA.py
            └── (other COCO annotation files, optional)
```

### Important

- `CNA.py` expects the COCO instances annotation at:

  ```python
  COCO_ANN_PATH = "./data/coco/annotations/instances_train2017.json"
  ```

- If your COCO annotations are in a different location, you **must** update `COCO_ANN_PATH` inside `CNA.py` to match your actual path.

For this part of the pipeline, only the **annotation file** is required. COCO images are not used yet (they become useful later when adding diffusion or SAM).

---

## 2. Environment & Dependencies

### 2.1 Python Environment

Use Python 3.9+ (Python 3.10 recommended). Example using conda (optional):

```bash
conda create -n dreammask_nss python=3.10 -y
conda activate dreammask_nss
```

### 2.2 Required Libraries

Both `CNA.py` and `batch_context_layout_generator.py` require:

- `openai` — LLM calls
- `beautifulsoup4` — HTML parsing
- `pillow` — layout image rendering

Install via:

```bash
pip install openai beautifulsoup4 pillow
```

> If you later add diffusion models or SAM, you will also need `torch` and the corresponding model libraries. They are **not required** for CNA + layout generation.

---

## 3. OpenAI API Key Setup

Both scripts use the OpenAI API. You must set the environment variable `OPENAI_API_KEY`.

### Linux / macOS

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Windows (PowerShell)

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

Restart your terminal after setting the variable so that Python can see it.

---

## 4. Script Overview

### 4.1 `CNA.py` — COCO-based Category Name Association

**Responsibilities:**

1. Load all category names from the COCO instances annotation file:

   ```python
   COCO_ANN_PATH = "./data/coco/annotations/instances_train2017.json"
   ```

   This becomes your **C_train** (base training vocabulary).

2. Use an LLM (e.g. `gpt-4.1-mini`) to generate **novel categories** that:
   - Are semantically related to COCO categories
   - Frequently co-occur with them in realistic scenes

3. Perform multiple sampling rounds and keep only **stable** novel categories:
   - A category must appear at least twice across LLM responses
   - It must not already be in the COCO base category list

4. Save results as:

   - `coco_categories.json` — COCO base categories (C_train)
   - `novel_categories.json` — generated novel categories (C_novel)

---

### 4.2 `batch_context_layout_generator.py` — Batch Context-aware Layout Generation

**Responsibilities:**

1. Load vocabulary:

   - `coco_categories.json` → **C_train**
   - `novel_categories.json` → **C_novel**

2. Build combined vocabulary:

   ```text
   C_all = C_train ∪ C_novel
   ```

3. For each synthetic scene (scene index `i`):

   - Randomly sample `min_objects–max_objects` categories from `C_all`
   - Ask GPT-5 (e.g. `gpt-5.1`) to design a **realistic 2D layout** as raw HTML using absolutely positioned `<div>` elements, e.g.:

     ```html
     <div class="object"
          data-name="car"
          style="position:absolute; left:100px; top:350px; width:220px; height:120px;">
     </div>
     ```

   - Parse every `<div class="object">` and extract:

     - `name`
     - `x`, `y`, `w`, `h` (in pixels)

   - Render a layout PNG with colored rectangles and text labels.

   - Save per-scene artifacts:
     - HTML → `scene_00000.html`
     - PNG → `scene_00000.png`
     - JSON metadata → `scene_00000.json`

---

## 5. How to Run

### 5.1 Step 1 — Generate COCO + Novel Categories (`CNA.py`)

1. Ensure the COCO instances annotation file exists at:

   ```text
   ./data/coco/annotations/instances_train2017.json
   ```

   or modify `COCO_ANN_PATH` in `CNA.py` to your actual path.

2. Ensure `OPENAI_API_KEY` is set (see Section 3).

3. Run:

   ```bash
   python CNA.py
   ```

Expected output includes messages such as:

- `[INFO] Loading COCO categories from ./data/coco/annotations/instances_train2017.json ...`
- `[INFO] Loaded XX base categories from COCO.`
- `[INFO] Generating novel categories with LLM...`
- `[INFO] Generated YY novel categories.`
- `[INFO] Saved ... categories to coco_categories.json`
- `[INFO] Saved ... categories to novel_categories.json`
- `[INFO] Done.`

After running, you should have:

```text
project_root/
  CNA.py
  batch_context_layout_generator.py
  coco_categories.json
  novel_categories.json
  README.md
  data/
    coco/
      annotations/instances_train2017.json
      ...
```

---

### 5.2 Step 2 — Batch-generate Context-aware Layouts

With `coco_categories.json` and `novel_categories.json` in the same folder as `batch_context_layout_generator.py`, run:

```bash
python batch_context_layout_generator.py
```

The script will:

1. Load C_train and C_novel
2. Build `C_all`
3. For each scene:
   - Sample a subset of categories
   - Call GPT-5.1 to generate an HTML layout
   - Parse bounding boxes
   - Render a PNG layout image
   - Save metadata

By default (inside `batch_context_layout_generator.py`), the main entry might look like:

```python
generate_batch_layouts(
    coco_categories_path="coco_categories.json",
    novel_categories_path="novel_categories.json",
    out_dir="layouts_dataset",
    num_scenes=50,
    canvas_width=1024,
    canvas_height=1024,
    min_objects=5,
    max_objects=8,
)
```

This will create:

```text
layouts_dataset/
  scene_00000.html
  scene_00000.png
  scene_00000.json
  scene_00001.html
  scene_00001.png
  scene_00001.json
  ...
```

Each `scene_XXXXX.json` file contains, for example:

```json
{
  "scene_index": 0,
  "scene_categories": [
    "car",
    "traffic light",
    "person",
    "bicycle",
    "bus"
  ],
  "boxes": [
    { "name": "car", "x": 120, "y": 430, "w": 260, "h": 140 },
    { "name": "person", "x": 300, "y": 350, "w": 80, "h": 200 }
  ],
  "html_path": "/absolute/path/to/layouts_dataset/scene_00000.html",
  "png_path": "/absolute/path/to/layouts_dataset/scene_00000.png",
  "canvas_width": 1024,
  "canvas_height": 1024
}
```

---

## 6. Tunable Parameters

**In `CNA.py`** (inside the call to `generate_related_categories`):

- `n_repeat` — number of LLM calls per prompt (e.g. 5–10)
- `top_k` — number of candidate categories per call (e.g. 30–50)

**In `batch_context_layout_generator.py`** (in the `generate_batch_layouts` call):

- `num_scenes` — how many synthetic layouts you want to generate
- `canvas_width`, `canvas_height` — resolution of the layout canvas
- `min_objects`, `max_objects` — minimum / maximum number of objects per scene

You can increase `num_scenes` (e.g. 500, 1000) to build a larger synthetic layout dataset.



## 7. Layout-to-image Generation

Once we have obtained the layout results, Please refer to 
LayoutGPT (https://github.com/weixi-feng/LayoutGPT) for realistic image generation. We assume the constructed dataset is shown as:


```text
layouts_dataset/
  scene_00000.html
  scene_00000.png
  real_00000.png
  scene_00000.json
  scene_00001.html
  scene_00001.png
  real_00001.png
  scene_00001.json
  ...
```


## 8. Troubleshooting

- **`FileNotFoundError: instances_train2017.json not found`**  
  - Check that COCO is under `./data/coco/annotations/instances_train2017.json`,  
    or update `COCO_ANN_PATH` in `CNA.py`.

- **`OPENAI_API_KEY` not set`**  
  - Set it as shown in Section 3 and restart your shell.

- **No objects parsed from HTML**  
  - Open `scene_XXXXX.html` to inspect the raw GPT output.  
  - If the format is off, adjust the prompt in `generate_layout_html_with_context`.

---

## 9. Mask Generation & Score-based Selection

After layout-to-image generation (e.g., using LayoutGPT), we assume your dataset
looks like this:

```text
layouts_dataset/
  scene_00000.html
  scene_00000.png        # layout visualization
  real_00000.png         # realistic synthesized image
  scene_00000.json       # metadata with bounding boxes
  scene_00001.html
  scene_00001.png
  real_00001.png
  scene_00001.json
  ...
```

The file `mask_and_selection.py` performs:

1. **Mask generation (SAM-based)**  
   - For each scene JSON in `layouts_dataset/`, it reads the corresponding `real_XXXXX.png`.  
   - For each object bounding box in `scene_XXXXX.json`, it uses **Segment Anything (SAM)**  
     to predict an instance mask.

2. **Score-based selection (CLIP + mask uncertainty)**  
   - Computes a CLIP image–text similarity score between the cropped object region  
     and its category name.  
   - Uses the SAM mask score as a simple proxy for **mask uncertainty**  
     (`mask_uncertainty = 1 - best_mask_score`).  
   - Keeps only objects that satisfy both:
     - `clip_score >= CLIP_THRESHOLD`  
     - `mask_uncertainty <= UNCERTAINTY_THRESHOLD`

3. **Processed dataset export**  
   - Saves the filtered masks and metadata under `processed_dataset/`:

     ```text
     processed_dataset/
       images/
         real_00000.png
         real_00001.png
         ...
       masks/
         scene_00000_obj_00.png
         scene_00000_obj_01.png
         ...
       scene_00000.json
       scene_00001.json
       ...
     ```

   - Each `scene_XXXXX.json` in `processed_dataset/` contains:
     - `scene_id`  
     - `real_image` (path to the copied `real_XXXXX.png`)  
     - `canvas_width`, `canvas_height` (if present in original metadata)  
     - `objects`: a list of selected objects, each with:
       - `index` (original object index)  
       - `name` (category name)  
       - `box` (`x, y, w, h`)  
       - `mask_path` (relative path to the saved mask PNG)  
       - `clip_score`  
       - `mask_uncertainty`  

### 9.1 Dependencies for `mask_and_selection.py`

Additional libraries required:

```bash
pip install torch torchvision
pip install opencv-python numpy pillow
pip install clip-anytorch
pip install git+https://github.com/facebookresearch/segment-anything.git
```

You also need to download a **SAM checkpoint**, for example `sam_vit_h_4b8939.pth`,
and place it under:

```text
./weights/sam_vit_h_4b8939.pth
```

This path is configured near the top of `mask_and_selection.py` as:

```python
SAM_CHECKPOINT = "./weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
```

Adjust these if you use a different SAM variant or path.

### 9.2 How to Run `mask_and_selection.py`

1. Make sure your `layouts_dataset/` has the structure shown above and that
   `scene_XXXXX.json` contains a `boxes` field with items of the form:

   ```json
   {
     "name": "car",
     "x": 120,
     "y": 430,
     "w": 260,
     "h": 140
   }
   ```

2. Verify that you have downloaded the SAM checkpoint into `./weights/`.

3. Install the additional dependencies listed in Section 7.1.

4. Run:

   ```bash
   python mask_and_selection.py
   ```

5. After it finishes, you will get a processed dataset under:

   ```text
   processed_dataset/
     images/
     masks/
     scene_00000.json
     scene_00001.json
     ...
   ```

   This dataset contains only high-quality object instances (according to the
   CLIP and mask-uncertainty thresholds) and is suitable for training
   the downstream segmentation model in the DreamMask-style pipeline.
