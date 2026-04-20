"""
review_classify.py

Goes through all classified images one by one for manual review.
Shows the processed image AND plays the matching video side by side.
Allows reassigning the class, which moves the image and updates the JSON.

Controls:
    1–6  → reassign to that class
    n    → confirm current label and move to next
    q    → save and quit

Usage:
    Set the liquid name below and run:
    python review_classify.py
    pip install fastai opencv-python iPython
"""

import os
import re
import json
import shutil
import tempfile
import cv2
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
liquid = "DMF"
base = Path(r"C:\Users\HV\Desktop\bruno_work\save_electrospray")
json_folder = base / liquid / "Current"
images_folder = base / liquid / "PROCESSED CLIPS"
clips_folder = base / liquid / "SPLIT CLIPS"
output_base = base / liquid / "CLASSIFIED"


# If json_files is empty, the script will load all available JSON files in json_folder.
# Otherwise specify the exact filenames to review, e.g. ["experiment_0.json", "experiment_1.json"].
JSON_FILES   = ['experiment_10.json', 'experiment_11.json']

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
print(f"[REVIEW] Using JSON files: {json_list}")


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
    print(f"[REVIEW] No classified images found in {output_base}")
    exit()

print(f"[REVIEW] Found {len(all_images)} images to review")
print(f"\nControls:")
for i, cls in enumerate(CLASSES, 1):
    print(f"  {i} → {cls}")
print("  n → confirm    q → save and quit\n")

PANEL_W      = 320
TARGET_H     = 480   # resize both image and video frame to this height

def make_panel(lines: list, height: int) -> np.ndarray:
    panel = np.zeros((height, PANEL_W, 3), dtype=np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return panel

def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    ratio = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * ratio), h))

# ── Review loop ───────────────────────────────────────────────────────
for idx, (img_path, current_class) in enumerate(all_images):

    match = re.match(r"clip_(\d+)_(\d+)\.(jpg|png)", img_path.name)
    if not match:
        print(f"[SKIP] Unexpected filename: {img_path.name}")
        continue

    experiment_idx = int(match.group(1))
    sample_idx     = int(match.group(2))
    sample_key     = f"sample {sample_idx}"

    result = load_json(experiment_idx)
    if result is None:
        print(f"[SKIP] No JSON for experiment {experiment_idx}")
        continue
    json_path, data = result
    sample_data = data[sample_key]

    # Matching video path
    video_stem = img_path.stem                          # e.g. clip_0_11
    video_path = clips_folder / (video_stem + ".mp4")
    cap        = cv2.VideoCapture(str(video_path)) if video_path.exists() else None
    fps        = cap.get(cv2.CAP_PROP_FPS) if cap else 25
    wait_ms    = max(1, int(1000 / fps))

    if cap is None:
        print(f"  [INFO] No video found for {img_path.name}, showing image only")

    # Static processed image (resized)
    static_img = cv2.imread(str(img_path))
    if static_img is None:
        print(f"[SKIP] Could not read image: {img_path.name}")
        if cap:
            cap.release()
        continue
    static_img = resize_to_height(static_img, TARGET_H)

    current_class = img_path.parent.name
    image_classification = sample_data.get("image_classification", current_class)
    voltage = sample_data.get("voltage", "N/A")
    flow_rate = sample_data.get("flow_rate", "N/A")

    info_lines = [
        (f"[{idx}/{len(all_images)}]",          (200, 200, 200)),
        (f"{img_path.name}",                       (200, 200, 200)),
        ("",                                        (0, 0, 0)),
        (f"Folder: {current_class}",               (0, 255, 0)),
        (f"JSON:   {image_classification}",       (0, 255, 255)),
        ("",                                        (0, 0, 0)),
        (f"voltage:   {voltage}",                 (200, 200, 200)),
        (f"flow_rate: {flow_rate}",               (200, 200, 200)),
        ("",                                        (0, 0, 0)),
    ] + [(f"{i+1}: {cls}", (180, 180, 180)) for i, cls in enumerate(CLASSES)] + [
        ("",                                        (0, 0, 0)),
        ("n: confirm",                             (180, 180, 180)),
        ("q: save & quit",                         (180, 180, 180)),
    ]

    decided = False
    while not decided:

        # Get next video frame (loops)
        video_frame = None
        if cap and cap.isOpened():
            ret, video_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, video_frame = cap.read()
            if ret:
                video_frame = resize_to_height(video_frame, TARGET_H)

        panel = make_panel(info_lines, TARGET_H)

        if video_frame is not None:
            display = np.hstack([panel, static_img, video_frame])
            cv2.setWindowTitle("Review", "Review  |  LEFT: processed image    RIGHT: video")
        else:
            display = np.hstack([panel, static_img])
            cv2.setWindowTitle("Review", "Review  |  (no video found)")

        cv2.imshow("Review", display)
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            if cap:
                cap.release()
            for exp_idx in json_cache:
                save_json(exp_idx)
            cv2.destroyAllWindows()
            print("\n[REVIEW] Quit — all changes saved.")
            exit()

        elif key == ord('n'):
            if image_classification != current_class:
                sample_data["image_classification"] = current_class
                save_json(experiment_idx)
                print(f"  [{idx}] Confirmed: {current_class} (JSON updated)")
            else:
                print(f"  [{idx}] Confirmed: {current_class}")
            decided = True

        elif key in [ord(str(i)) for i in range(1, len(CLASSES) + 1)]:
            new_class = CLASSES[int(chr(key)) - 1]

            if new_class == current_class:
                if image_classification != current_class:
                    sample_data["image_classification"] = current_class
                    save_json(experiment_idx)
                    print(f"  [{idx}] Confirmed: {current_class} (JSON updated)")
                else:
                    print(f"  [{idx}] Same class confirmed: {current_class}")
            else:
                new_folder = output_base / new_class
                os.makedirs(new_folder, exist_ok=True)
                shutil.move(str(img_path), str(new_folder / img_path.name))
                sample_data["image_classification"] = new_class
                save_json(experiment_idx)
                print(f"  [{idx}] Changed: {current_class} → {new_class}")

            decided = True

    if cap:
        cap.release()

cv2.destroyAllWindows()
for exp_idx in json_cache:
    save_json(exp_idx)

print(f"\n[REVIEW] Done. All {len(all_images)} images reviewed.")