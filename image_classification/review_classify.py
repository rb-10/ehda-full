import os
import re
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
liquid = "DMF"
base = Path(r"C:\Users\HV\Desktop\bruno_work\main\data")
json_folder = base / liquid / "Current"
images_folder = base / liquid / "PROCESSED CLIPS"
clips_folder = base / liquid / "SPLIT CLIPS"
output_base = base / liquid / "CLASSIFIED"

JSON_FILES   = []
CLASSES = ["cone_jet", "dripping", "intermitent", "multi_jet", "unconclusive", "undefined"]

# ── JSON helpers ──────────────────────────────────────────────────────
json_cache = {}
dirty_jsons = set() # Keeps track of which JSONs were modified but not saved

def get_json_file_list():
    if JSON_FILES:
        return [f for f in JSON_FILES if (json_folder / f).exists()]
    return [p.name for p in sorted(json_folder.glob("*.json")) if p.is_file()]

json_list = get_json_file_list()

def load_json(experiment_idx: int):
    if experiment_idx in json_cache:
        return json_cache[experiment_idx]
    json_filename = f"experiment_{experiment_idx}.json"
    json_path = json_folder / json_filename
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    json_cache[experiment_idx] = (json_path, data)
    return json_cache[experiment_idx]

def save_all_dirty():
    """Writes modified data from RAM to Disk."""
    if not dirty_jsons:
        return
    print(f"\n[SAVING] Writing {len(dirty_jsons)} modified JSON(s) to disk...")
    for exp_idx in list(dirty_jsons):
        json_path, data = json_cache[exp_idx]
        tmp = str(json_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp, json_path)
        dirty_jsons.remove(exp_idx)
    print("[SAVED] Disk sync complete.")

# ── Collect Images ────────────────────────────────────────────────────
all_images = []
for cls in CLASSES:
    folder = output_base / cls
    if not folder.exists(): continue
    for img_path in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")):
        all_images.append((img_path, cls))

all_images.sort(key=lambda x: x[0].name)

PANEL_W, TARGET_H = 320, 480

def make_panel(lines: list, height: int) -> np.ndarray:
    panel = np.zeros((height, PANEL_W, 3), dtype=np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return panel

def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    ratio = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * ratio), h))

# ── Review loop ───────────────────────────────────────────────────────
print(f"[REVIEW] Starting review for {len(all_images)} images...")

for idx, (img_path, folder_class) in enumerate(all_images):
    match = re.match(r"clip_(\d+)_(\d+)\.(jpg|png)", img_path.name)
    if not match: continue

    exp_idx, smp_idx = int(match.group(1)), int(match.group(2))
    sample_key = f"sample {smp_idx}"

    result = load_json(exp_idx)
    if not result: continue
    json_path, data = result
    sample_data = data[sample_key]

    if "manual_classification" in sample_data:
        continue

    video_path = clips_folder / (img_path.stem + ".mp4")
    cap = cv2.VideoCapture(str(video_path)) if video_path.exists() else None
    fps = cap.get(cv2.CAP_PROP_FPS) if cap else 25
    wait_ms = max(1, int(1000 / fps))

    static_img = cv2.imread(str(img_path))
    if static_img is None: continue
    static_img = resize_to_height(static_img, TARGET_H)

    info_lines = [
        (f"FILE: {img_path.name}", (200, 200, 200)),
        (f"Auto-Label: {sample_data.get('image_classification', 'N/A')}", (0, 255, 255)),
        (f"V: {sample_data.get('voltage','N/A')} | Q: {sample_data.get('flow_rate','N/A')}", (200, 200, 200)),
        ("", (0, 0, 0)),
        ("SELECT MANUAL CLASS:", (255, 255, 255)),
    ] + [(f"{i+1}: {cls}", (180, 180, 180)) for i, cls in enumerate(CLASSES)] + [
        ("", (0, 0, 0)),
        ("n: confirm current folder", (0, 255, 0)),
        ("q: save & quit", (0, 0, 255)),
    ]

    decided = False
    while not decided:
        video_frame = None
        if cap and cap.isOpened():
            ret, video_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, video_frame = cap.read()
            if ret:
                video_frame = resize_to_height(video_frame, TARGET_H)

        panel = make_panel(info_lines, TARGET_H)
        display = np.hstack([panel, static_img, video_frame]) if video_frame is not None else np.hstack([panel, static_img])
        
        cv2.imshow("Manual Review", display)
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            if cap: cap.release()
            save_all_dirty() # Disk sync before exit
            cv2.destroyAllWindows()
            exit()

        elif key == ord('n'):
            sample_data["manual_classification"] = folder_class
            dirty_jsons.add(exp_idx) # Mark for later saving
            decided = True

        elif key in [ord(str(i)) for i in range(1, len(CLASSES) + 1)]:
            new_class = CLASSES[int(chr(key)) - 1]
            if new_class != folder_class:
                new_folder = output_base / new_class
                new_folder.mkdir(exist_ok=True)
                shutil.move(str(img_path), str(new_folder / img_path.name))
            
            sample_data["manual_classification"] = new_class
            dirty_jsons.add(exp_idx) # Mark for later saving
            decided = True
    
    # Save every 50 classifications just in case of a crash
    if len(dirty_jsons) > 0 and idx % 50 == 0:
        save_all_dirty()

    if cap: cap.release()

cv2.destroyAllWindows()
save_all_dirty()
print("Review Session Complete.")