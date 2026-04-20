"""
auto_classify.py

Uses the trained model to automatically classify all images in the
unclassified folder, moves them to the correct class subfolder,
and updates the spray_mode in the matching JSON experiment file.

Usage:
    Set the liquid name below and run:
    python auto_classify.py
"""

import os
import re
import json
import shutil
from pathlib import Path
from fastai.vision.all import load_learner, PILImage
import tempfile


# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH   = "final_model/export.pkl"
CONFIDENCE_THRESHOLD = 0.80   # below this → saved as "unconclusive"
RECLASSIFY_EXISTING = False  # True = re-run classification even if JSON already has image_classification
SOLUTION = "DMF"
JSON_FOLDER  = Path(rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\{SOLUTION}\Current")
INPUT_FOLDER = Path(rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\{SOLUTION}\PROCESSED CLIPS")
OUTPUT_BASE  = Path(rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\{SOLUTION}\CLASSIFIED")

CLASSES = ["cone_jet", "dripping", "intermitent", "multi_jet", "unconclusive"]

# ── Setup ─────────────────────────────────────────────────────────────
for cls in CLASSES:
    os.makedirs(OUTPUT_BASE / cls, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────
print(f"[AUTO] Loading model: {MODEL_PATH}")
learn = load_learner(MODEL_PATH)
print(f"[AUTO] Classes: {list(learn.dls.vocab)}\n")

# ── Load / save JSON helpers ──────────────────────────────────────────
json_cache = {}

def load_json(experiment_idx: int):
    if experiment_idx in json_cache:
        return json_cache[experiment_idx]
    json_path = JSON_FOLDER / f"experiment_{experiment_idx}.json"
    if not json_path.exists():
        print(f"  [WARNING] JSON not found: {json_path}")
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    json_cache[experiment_idx] = (json_path, data)
    return json_cache[experiment_idx]

def save_json(experiment_idx: int):
    if experiment_idx not in json_cache:
        return
    json_path, data = json_cache[experiment_idx]

    # Write to a temp file first, then rename — rename is atomic on all OS
    tmp_path = str(json_path) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, json_path)   # atomic — crash can't corrupt original

# ── Process images ────────────────────────────────────────────────────
images = sorted(INPUT_FOLDER.glob("*.jpg")) + \
         sorted(INPUT_FOLDER.glob("*.png"))

if not images:
    print(f"[AUTO] No images found in {INPUT_FOLDER}")
    exit()

print(f"[AUTO] Found {len(images)} images to classify\n")
print(f"{'Image':<40} {'Prediction':<20} {'Confidence':>10}")
print("─" * 72)

skipped = []

for img_path in images:

    match = re.match(r"clip_(\d+)_(\d+)\.(jpg|png)", img_path.name)
    if not match:
        print(f"  [SKIP] Unexpected filename format: {img_path.name}")
        skipped.append(img_path.name)
        continue

    experiment_idx = int(match.group(1))
    sample_idx     = int(match.group(2))
    sample_key     = f"sample {sample_idx}"


    result = load_json(experiment_idx)
    if result is None:
        skipped.append(img_path.name)
        continue

    json_path, data = result
    if sample_key not in data:
        print(f"  [SKIP] '{sample_key}' not in experiment_{experiment_idx}.json")
        skipped.append(img_path.name)
        continue


    # Check if already classified
    already_classified = "image_classification" in data[sample_key]
    if already_classified:
        previous_class = data[sample_key]["image_classification"]
        if not RECLASSIFY_EXISTING:
            chosen_class = previous_class
            print(f"  [SKIP] Already classified: {img_path.name} -> {chosen_class}")
            skipped.append(img_path.name)
            # Copy image to correct class folder
            dest = OUTPUT_BASE / chosen_class / img_path.name
            if not dest.exists():
                shutil.copy2(str(img_path), str(dest))
            continue
        else:
            print(f"  [RECLASSIFY] Reclassifying: {img_path.name} (was {previous_class})")

    # Run model
    img              = PILImage.create(img_path)
    label, _, probs  = learn.predict(img)
    confidence       = probs.max().item()

    # Map FastAI label to your class names — adjust if your training
    # labels differ from CLASSES (e.g. "Cone Jet" vs "cone_jet")
    chosen_class = str(label).lower().replace(" ", "_")

    # Fall back to unconclusive if confidence is too low
    if confidence < CONFIDENCE_THRESHOLD:
        chosen_class = "unconclusive"

    # Safety: make sure chosen_class is valid
    if chosen_class not in CLASSES:
        chosen_class = "unconclusive"

    # Update JSON: save classification in 'image_classification' field
    data[sample_key]["image_classification"] = chosen_class
    save_json(experiment_idx)


    # Copy image
    dest = OUTPUT_BASE / chosen_class / img_path.name
    shutil.copy2(str(img_path), str(dest))

    print(f"{img_path.name:<40} {chosen_class:<20} {confidence:>9.1%}")

# Final save of all modified JSONs
for exp_idx in json_cache:
    save_json(exp_idx)

print(f"\n[AUTO] Done. {len(images) - len(skipped)} classified, {len(skipped)} skipped.")
