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
BASE         = Path(r"data")

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

    # Count total rows for this video file in DB
    cursor_count = db._conn.execute("SELECT COUNT(*) FROM measurements WHERE video_file = ?", (original_video_name,))
    total_db_rows = cursor_count.fetchone()[0]

    # Count how many image clips exist for this video file across all output folders (or in CLIPS_FOLDER)
    # We can check how many clips exist for this base_video_part in CLIPS_FOLDER
    matching_clips = list(CLIPS_FOLDER.glob(f"clip_{base_video_part}_*.mp4"))
    num_clips = len(matching_clips) if len(matching_clips) > 0 else 1

    rows_per_clip = max(1, total_db_rows // num_clips)
    offset = clip_index * rows_per_clip

    # Fetch Metadata and check existing manual classification for rows of this clip segment
    query = """SELECT id, actual_voltage, flow_rate, image_classification, manual_classification, raw_data_file, sample_rate 
               FROM measurements WHERE video_file = ? 
               ORDER BY timestamp ASC LIMIT ? OFFSET ?"""
    cursor = db._conn.execute(query, (original_video_name, rows_per_clip, offset))
    rows = cursor.fetchall()
    if not rows:
        print(f"[SKIP] DB Record not found for {img_path.name}")
        continue

    # Use metadata from the first row in the segment for UI display
    db_id, voltage, flow, ai_label, manual_class, _, sample_rate = rows[0]
    segment_ids = [r[0] for r in rows]

    # Select the row with highest ID in this segment to get raw_data_file
    last_row = max(rows, key=lambda r: r[0])
    raw_data_file = last_row[5]

    # Render signal plot matching plot_raw_waveforms.py style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt
    import io

    LABEL_COLOURS = {
        'dripping': "#2e62d4",
        'intermitent': "#066400",
        'cone_jet': "#e80101",
        'multi_jet': "#830068",
    }
    DEFAULT_COLOUR = "#00E5FF"
    MULTIPLIER_NA = 1

    raw_waveforms_dir = BASE / "raw_waveforms"
    plot_img = np.zeros((TARGET_H, 480, 3), dtype=np.uint8)

    if raw_data_file:
        npy_path = raw_waveforms_dir / raw_data_file
        if not npy_path.exists() and not raw_data_file.endswith('.npy'):
            npy_path = raw_waveforms_dir / f"{raw_data_file}.npy"

        if npy_path.exists():
            try:
                raw_data = np.load(str(npy_path)) * MULTIPLIER_NA
                raw_data = np.squeeze(raw_data)

                sr = float(sample_rate) if sample_rate else 100000.0
                t_ms = np.arange(len(raw_data)) / sr * 1000.0

                show_filtered = sr >= 6000
                if show_filtered:
                    cutoff_norm = 1000.0 / (0.5 * sr)
                    b, a = butter(6, Wn=cutoff_norm, btype="low", analog=False)
                    filtered_data = filtfilt(b, a, raw_data)

                # Determine color based on current class or manual class
                eff_label = manual_class if manual_class in LABEL_COLOURS else current_class
                colour = LABEL_COLOURS.get(eff_label, DEFAULT_COLOUR)

                fig, ax = plt.subplots(figsize=(5.5, 5), dpi=100)
                fig.patch.set_facecolor("#FFFFFF")
                ax.set_facecolor('#FFFFFF')

                if show_filtered:
                    ax.plot(t_ms, raw_data, color=colour, alpha=0.35, linewidth=0.8, label="Raw")
                    ax.plot(t_ms, filtered_data, color=colour, alpha=0.95, linewidth=1.6, label="Filtered")
                else:
                    ax.plot(t_ms, raw_data, color=colour, alpha=0.9, linewidth=1.0, label="Raw")

                ax.set_title(f"Label: {eff_label} | ID: {db_id} | V: {voltage}V | Q: {flow}", color='black', fontsize=9)
                ax.set_xlabel("Time (ms)", color='black', fontsize=8)
                ax.set_ylabel("Current (nA)", color='black', fontsize=8)
                ax.tick_params(colors='black', labelsize=7)
                ax.grid(True, linestyle="--", alpha=0.3, color='#555555')
                for spine in ax.spines.values():
                    spine.set_color('#444444')
                ax.set_ylim((-20, 300))
                if show_filtered:
                    ax.legend(fontsize=7, loc="upper right", facecolor='#2a2a2a', edgecolor='#444444', labelcolor='black')

                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                
                # Convert PNG buffer to OpenCV BGR image
                file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
                mat = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if mat is not None:
                    plot_img = resize_to_height(mat, TARGET_H)
            except Exception as e:
                cv2.putText(plot_img, f"Error plotting signal: {e}", (10, TARGET_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            cv2.putText(plot_img, "Signal file not found", (10, TARGET_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(plot_img, "No raw_data_file in DB", (10, TARGET_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- SKIP LOGIC ---
    # Skip if manual_classification is already set to one of our valid classes for all rows in segment
    if manual_class in CLASSES:
        print(f"[{idx+1}] Skipping: Already classified as '{manual_class}'")
        #continue
    if ai_label == "multi_jet (100%)":
        print(f"[{idx+1}] Skipping: Multi jet 100%")
        #new_class = "multi_jet"
        ##for sid in segment_ids:
        ##    db._conn.execute("UPDATE measurements SET manual_classification = ? WHERE id = ?", (new_class, sid))
        #db._conn.commit()
        #continue
    if ai_label == "dripping (100%)":
        print(f"[{idx+1}] Skipping: dripping 100%")
        #new_class = "dripping"
        #for sid in segment_ids:
        #    db._conn.execute("UPDATE measurements SET manual_classification = ? WHERE id = ?", (new_class, sid))
        #db._conn.commit()
        #continue
    if idx + 1 < 1000:
        print(f"[{idx+1}] Skipping: Already classified as '{manual_class}'")
        continue
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
        (f"Rows/clip: {rows_per_clip}",  (200, 200, 200)),
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

        # Combined display: Metadata Panel | Processed Photo | Raw Video | Current Signal Plot
        display = np.hstack([panel, static_img, video_frame, plot_img])
        cv2.imshow("Review & Reclassify", display)
        
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            db.close()
            cv2.destroyAllWindows()
            exit()

        elif key == ord('n'):
            # 1. Clean the AI label (e.g., "dripping (76%)" -> "dripping")
            clean_ai_class = ai_label.split('(')[0].strip()
            
            # 2. Update DB for all rows belonging to this video clip
            db._conn.executemany("UPDATE measurements SET manual_classification = ? WHERE id = ?", [(clean_ai_class, sid) for sid in segment_ids])
            db._conn.commit()
            
            print(f"[{idx+1}] Confirmed AI class: {clean_ai_class} for {len(segment_ids)} DB rows")
            decided = True

        elif key in [ord(str(i)) for i in range(1, len(CLASSES) + 1)]:
            new_class = CLASSES[int(chr(key)) - 1]
            
            if new_class != current_class:
                # 1. Update Database manual classification for all rows belonging to this clip
                db._conn.executemany("UPDATE measurements SET manual_classification = ? WHERE id = ?", [(new_class, sid) for sid in segment_ids])
                db._conn.commit()
                
                # 2. Move file to new class folder
                new_folder = OUTPUT_BASE / new_class
                new_folder.mkdir(exist_ok=True)
                shutil.move(str(img_path), str(new_folder / img_path.name))
                
                print(f"[{idx+1}] Reclassified & Moved ({len(segment_ids)} DB rows updated): {current_class} -> {new_class}")
            else: 
                # If the chosen class is the same as the folder, still save it to the DB
                # but ensure we strip any percentages if they somehow exist in folder_class
                clean_current_class = current_class.split('(')[0].strip()
                db._conn.executemany("UPDATE measurements SET manual_classification = ? WHERE id = ?", [(clean_current_class, sid) for sid in segment_ids])
                db._conn.commit()
                print(f"[{idx+1}] Folder class confirmed and saved for {len(segment_ids)} DB rows: {clean_current_class}")

            decided = True
            


    cap.release()

db.close()
cv2.destroyAllWindows()
print("\n[REVIEW] Finished reviewing all images.")