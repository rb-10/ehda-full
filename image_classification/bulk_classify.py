"""
bulk_classify.py

Bulk update image classifications for a range of items.
Updates JSON and moves images to the correct class folders.

Usage:
    python bulk_classify.py

    Then follow the prompts to specify:
    - Start index (1-based)
    - End index (1-based)
    - New class

Example:
    Start index: 1521
    End index: 1550
    New class: intermitent
"""

import os
import re
import json
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
liquid = "Ethanol"
base = Path(r"C:\Users\Bruno Duarte\Documents\datasets")
json_folder = base / liquid / "Current"
images_folder = base / liquid / "PROCESSED CLIPS"
clips_folder = base / liquid / "SPLIT CLIPS"
output_base = base / liquid / "CLASSIFIED"

# If json_files is empty, the script will load all available JSON files in json_folder.
# Otherwise specify the exact filenames to review, e.g. ["experiment_0.json", "experiment_1.json"].
JSON_FILES   = ['experiment_16.json', 'experiment_17.json']

CLASSES = ["cone_jet", "dripping", "intermitent", "multi_jet", "unconclusive", "undefined"]

# ── JSON helpers (atomic write) ───────────────────────────────────────
json_cache = {}

def get_json_file_list():
    if JSON_FILES:
        selected = []
        missing = []
        for file_name in JSON_FILES:
            if (json_folder / file_name).exists():
                selected.append(file_name)
            else:
                missing.append(file_name)
        if missing:
            print(f"[WARNING] Missing JSON files: {missing}")
        return selected
    return [p.name for p in sorted(json_folder.glob("*.json")) if p.is_file()]

json_list = get_json_file_list()
print(f"[BULK] Using JSON files: {json_list}")

def load_json(experiment_idx: int):
    if experiment_idx in json_cache:
        return json_cache[experiment_idx]
    json_filename = f"experiment_{experiment_idx}.json"
    if json_filename not in json_list:
        return None
    json_path = json_folder / json_filename
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    json_cache[experiment_idx] = (json_path, data)
    return json_cache[experiment_idx]

def save_json(experiment_idx: int):
    if experiment_idx not in json_cache:
        return
    json_path, data = json_cache[experiment_idx]
    tmp = str(json_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, json_path)   # atomic — crash-safe

# ── Load all JSONs upfront ──────────────────────────────────────────
print(f"[BULK] Loading JSON files...")
for json_filename in json_list:
    match = re.match(r"experiment_(\d+)\.json", json_filename)
    if match:
        experiment_idx = int(match.group(1))
        load_json(experiment_idx)  # This populates the cache

print(f"[BULK] Loaded {len(json_cache)} JSON files into cache")

# ── Collect all classified images ─────────────────────────────────────
all_images = []
for cls in CLASSES:
    folder = output_base / cls
    if not folder.exists():
        continue
    for img_path in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")):
        all_images.append((img_path, cls))

all_images.sort(key=lambda x: x[0].name)

if not all_images:
    print(f"[BULK] No classified images found in {output_base}")
    exit()

print(f"[BULK] Found {len(all_images)} images total")

# ── Bulk update ───────────────────────────────────────────────────────
print("\nAvailable classes:")
for i, cls in enumerate(CLASSES, 1):
    print(f"  {i}: {cls}")

try:
    start_idx = int(input("\nStart index (1-based): ")) - 1  # Convert to 0-based
    end_idx = int(input("End index (1-based): ")) - 1      # Convert to 0-based
    class_choice = int(input("Class number (1-6): ")) - 1
    new_class = CLASSES[class_choice]
except (ValueError, IndexError):
    print("[ERROR] Invalid input. Exiting.")
    exit()

if start_idx < 0 or end_idx >= len(all_images) or start_idx > end_idx:
    print(f"[ERROR] Invalid range. Total images: {len(all_images)}")
    exit()

print(f"\n[BULK] Updating images {start_idx+1} to {end_idx+1} to class '{new_class}'...")

updated_count = 0
for idx in range(start_idx, end_idx + 1):
    img_path, current_class = all_images[idx]

    match = re.match(r"clip_(\d+)_(\d+)\.(jpg|png)", img_path.name)
    if not match:
        print(f"[SKIP] Unexpected filename: {img_path.name}")
        continue

    experiment_idx = int(match.group(1))
    sample_idx = int(match.group(2))
    sample_key = f"sample {sample_idx}"

    if experiment_idx not in json_cache:
        print(f"[SKIP] No JSON for experiment {experiment_idx}")
        continue
    json_path, data = json_cache[experiment_idx]
    sample_data = data[sample_key]

    if new_class != current_class:
        # Move image to new folder
        new_folder = output_base / new_class
        os.makedirs(new_folder, exist_ok=True)
        shutil.move(str(img_path), str(new_folder / img_path.name))
        print(f"  [{idx+1}] Moved: {current_class} → {new_class}")

    # Update JSON in cache
    sample_data["image_classification"] = new_class
    updated_count += 1

# ── Save all updated JSONs ────────────────────────────────────────────
print(f"\n[BULK] Saving {len(json_cache)} JSON files...")
for exp_idx in json_cache:
    save_json(exp_idx)

print(f"\n[BULK] Done. Updated {updated_count} images.")