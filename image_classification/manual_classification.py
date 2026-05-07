import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Goes up 3 levels to 'main/'
sys.path.insert(0, str(project_root))

import shutil
import cv2
import numpy as np
from pathlib import Path
from mapping.software.database import ElectrosprayDatabase

# ── Config ────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\HV\Desktop\bruno_work\main\data")

# Folder Structure
CLIPS_FOLDER = BASE / "images" / "SPLIT" / "SPLIT CLIPS"
IMAGES_FOLDER = BASE / "images" / "PROCESSED CLIPS"
OUTPUT_BASE  = BASE / "images" / "CLASSIFIED"

CLASSES = ["cone_jet", "dripping", "intermitent", "multi_jet", "unconclusive", "undefined"]

# ── Database Setup ───────────────────────────────────────────────────
db = ElectrosprayDatabase(str(BASE))

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

# ── UI Helpers ────────────────────────────────────────────────────────
PANEL_W  = 320
TARGET_H = 480 

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
print(f"[REVIEW] Found {len(all_images)} images to review")

for idx, (img_path, current_class) in enumerate(all_images):
    # Parse filename for DB lookup
    try:
        clean_name = img_path.stem.replace("clip_", "")
        base_video_part, index_part = clean_name.rsplit('_', 1)
        original_video_name = base_video_part + ".mp4"
        clip_index = int(index_part)
    except:
        print(f"[SKIP] Filename error: {img_path.name}")
        continue

    # Fetch Metadata and check existing manual classification
    query = """SELECT id, actual_voltage, flow_rate, image_classification, manual_classification 
               FROM measurements WHERE video_file = ? 
               ORDER BY timestamp ASC LIMIT 1 OFFSET ?"""
    cursor = db._conn.execute(query, (original_video_name, clip_index))
    row = cursor.fetchone()
    if not row:
        print(f"[SKIP] DB Record not found for {img_path.name}")
        continue

    db_id, voltage, flow, ai_label, manual_class = row
    # --- SKIP LOGIC ---
    # Skip if manual_classification is already set to one of our valid classes
    if manual_class in CLASSES:
        print(f"[{idx+1}] Skipping: Already classified as '{manual_class}'")
        continue
    if idx + 1 < 3360:
        print(f"[{idx+1}] Skipping: Already classified as '{manual_class}'")
        #continue
    # Video setup
    video_path = CLIPS_FOLDER / (img_path.stem + ".mp4")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25
    wait_ms = max(1, int(1000 / fps))

    # Processed static image
    static_img = cv2.imread(str(img_path))
    if static_img is not None:
        static_img = resize_to_height(static_img, TARGET_H)
    else:
        static_img = np.zeros((TARGET_H, TARGET_H, 3), np.uint8)

    info_lines = [
        (f"[{idx+1}/{len(all_images)}]", (200, 200, 200)),
        (f"V: {voltage}V | Q: {flow}",   (200, 200, 200)),
        ("", (0, 0, 0)),
        (f"Folder: {current_class}",     (0, 255, 0)),
        (f"Image Model: {ai_label}",              (0, 200, 255)),
        (f"Manual Class: {manual_class}",              (0, 200, 255)),
        ("", (0, 0, 0)),
    ] + [(f"{i+1}: {cls}", (180, 180, 180)) for i, cls in enumerate(CLASSES)] + [
        ("", (0, 0, 0)),
        ("n: confirm/next", (150, 150, 150)),
        ("q: quit",         (150, 150, 150))
    ]

    decided = False
    while not decided:
        ret, frame = cap.read()
        if not ret and cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        video_frame = resize_to_height(frame, TARGET_H) if ret else np.zeros((TARGET_H, 10, 3), np.uint8)
        panel = make_panel(info_lines, TARGET_H)

        # Combined display: Metadata Panel | Processed Photo | Raw Video
        display = np.hstack([panel, static_img, video_frame])
        cv2.imshow("Review & Reclassify", display)
        
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            db.close()
            cv2.destroyAllWindows()
            exit()

        elif key == ord('n'):
            # 1. Clean the AI label (e.g., "dripping (76%)" -> "dripping")
            clean_ai_class = ai_label.split('(')[0].strip()
            
            # 2. Update DB with the cleaned AI class
            db._conn.execute("UPDATE measurements SET manual_classification = ? WHERE id = ?", (clean_ai_class, db_id))
            db._conn.commit()
            
            print(f"[{idx+1}] Confirmed AI class: {clean_ai_class}")
            decided = True

        elif key in [ord(str(i)) for i in range(1, len(CLASSES) + 1)]:
            new_class = CLASSES[int(chr(key)) - 1]
            
            if new_class != current_class:
                # 1. Update Database manual classification
                db._conn.execute("UPDATE measurements SET manual_classification = ? WHERE id = ?", (new_class, db_id))
                db._conn.commit()
                
                # 2. Move file to new class folder
                new_folder = OUTPUT_BASE / new_class
                new_folder.mkdir(exist_ok=True)
                shutil.move(str(img_path), str(new_folder / img_path.name))
                
                print(f"[{idx+1}] Reclassified & Moved: {current_class} -> {new_class}")
            else: 
                # If the chosen class is the same as the folder, still save it to the DB
                # but ensure we strip any percentages if they somehow exist in folder_class
                clean_current_class = current_class.split('(')[0].strip()
                db._conn.execute("UPDATE measurements SET manual_classification = ? WHERE id = ?", (clean_current_class, db_id))
                db._conn.commit()
                print(f"[{idx+1}] Folder class confirmed and saved: {clean_current_class}")

            decided = True
            


    cap.release()

db.close()
cv2.destroyAllWindows()
print("\n[REVIEW] Finished reviewing all images.")