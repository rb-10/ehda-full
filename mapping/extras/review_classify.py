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
"""

import os
import re
import json
import shutil
import tempfile
import cv2
import numpy as np
from pathlib import Path
from fastai.vision.all import load_learner, PILImage
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────
LIQUID       = "EWG343"
MODEL_PATH   = "models/export.pkl"

BASE         = Path("C:/Users/HV/Desktop/bruno-mapping/dataset")
JSON_FOLDER  = BASE / "current"  / LIQUID / "classified"
OUTPUT_BASE  = BASE / "processed_images"   / LIQUID
VIDEO_FOLDER = BASE / "images"   / LIQUID / "unclassified"  # videos stay here, never moved

CLASSES = ["cone_jet", "dripping", "intermitent", "multi_jet", "EXCLUDE", "undefined"]

# ── Load model ────────────────────────────────────────────────────────
print(f"[REVIEW] Loading model: {MODEL_PATH}")
learn = load_learner(MODEL_PATH)

# ── JSON helpers (atomic write) ───────────────────────────────────────
json_cache = {}

def load_json(experiment_idx: int):
    if experiment_idx in json_cache:
        return json_cache[experiment_idx]
    json_path = JSON_FOLDER / f"experiment_{experiment_idx}.json"
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
    folder = OUTPUT_BASE / cls
    if not folder.exists():
        continue
    for img_path in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")):
        all_images.append((img_path, cls))

all_images.sort(key=lambda x: x[0].name)

if not all_images:
    print(f"[REVIEW] No classified images found in {OUTPUT_BASE}")
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


from scipy.signal import butter, filtfilt

def lowpass_filter(signal, fs, cutoff=3000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)

    return filtered

def make_current_plot(current_values, sample_idx=None, height=480, width=500):
    if not current_values or len(current_values) < 2:
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    dpi   = 100
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    vals = np.array(current_values, dtype=float)
    vals_filtered = lowpass_filter(vals, 100000, cutoff=3000)

    ax.plot(vals_filtered, linewidth=1, color='steelblue')

    # Correct label — sample_idx is the experiment step number, not a waveform index
    if sample_idx is not None:
        ax.text(0.02, 0.95,
                f"Sample: {sample_idx}  |  N={len(vals):,}pts",
                transform=ax.transAxes,
                fontsize=9, verticalalignment='top')

    # Remove the wrong axvline — sample_idx is not a waveform position
    # Optional: mark the midpoint of the acquisition instead
    ax.axvline(len(vals) // 2, color='gray', linestyle='--',
               linewidth=0.8, alpha=0.5, label='midpoint')

    ax.set_xlabel("Sample index (@ 100 kHz)")
    ax.set_ylabel("Current") 
    ax.set_ylim(-300, 3500)           
    ax.set_title("Current waveform")
    plt.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img



# ── Review loop ───────────────────────────────────────────────────────
for idx, (img_path, current_class) in enumerate(all_images):
    if(idx+1 < 232):
        print(f"  [{idx+1}] Skipped")
        continue

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
    sample_data = data.get(sample_key, {})
    current_values = sample_data.get("current", [])
    current_plot = make_current_plot(current_values, sample_idx, TARGET_H, 500)
    # Model prediction
    pil_img         = PILImage.create(img_path)
    label, _, probs = learn.predict(pil_img)
    confidence      = probs.max().item()
    model_label     = str(label).lower().replace(" ", "_")

    # Matching video path
    video_stem = img_path.stem                          # e.g. clip_0_11
    video_path = VIDEO_FOLDER / (video_stem + ".mp4")
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

    info_lines = [
        (f"[{idx+1}/{len(all_images)}]",          (200, 200, 200)),
        (f"{img_path.name}",                       (200, 200, 200)),
        ("",                                        (0, 0, 0)),
        (f"Current: {current_class}",              (0, 255, 0)),
        (f"Model:   {model_label}",                (0, 200, 255)),
        (f"Conf:    {confidence:.1%}",             (0, 200, 255)),
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
            display = np.hstack([panel, static_img, video_frame, current_plot])
            cv2.setWindowTitle("Review", "Review  |  LEFT: processed image    RIGHT: video")
        else:
            display = np.hstack([panel, static_img, current_plot])
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
            print(f"  [{idx+1}] Confirmed: {current_class}")
            decided = True

        elif key in [ord(str(i)) for i in range(1, len(CLASSES) + 1)]:
            new_class = CLASSES[int(chr(key)) - 1]

            if new_class == current_class:
                print(f"  [{idx+1}] Same class confirmed: {current_class}")
            else:
                new_folder = OUTPUT_BASE / new_class
                os.makedirs(new_folder, exist_ok=True)
                shutil.move(str(img_path), str(new_folder / img_path.name))
                data[sample_key]["spray_mode"] = new_class
                save_json(experiment_idx)
                print(f"  [{idx+1}] Changed: {current_class} → {new_class}")

            decided = True

    if cap:
        cap.release()

cv2.destroyAllWindows()
for exp_idx in json_cache:
    save_json(exp_idx)

print(f"\n[REVIEW] Done. All {len(all_images)} images reviewed.")