import cv2
import os
import shutil

# Import your database class
from mapping.software.database import ElectrosprayDatabase

base_data_path = r"C:\Users\HV\Desktop\bruno_work\main\data"
liquid = "EW82_HV_nz_21-04" 

# Paths
clips_folder = os.path.join(base_data_path, liquid, "SPLIT CLIPS")
images_folder = os.path.join(base_data_path, liquid, "PROCESSED CLIPS")
output_base = os.path.join(base_data_path, liquid, "CLASSIFIED")

classes = ["cone_jet", "dripping", "intermitent", "multi_jet", "unconclusive", "undefined"]
for cls in classes:
    os.makedirs(os.path.join(output_base, cls), exist_ok=True)

# Connect to Database (assuming data.db is in the base_data_path)
db = ElectrosprayDatabase(base_data_path)

videos = [f for f in os.listdir(clips_folder) if f.endswith(".mp4")]
videos.sort()

print("Controls:")
print("1–6 → assign class")
print("q   → quit")
print("n   → skip video")

def ensure_classified_copy(video_path, chosen_class, current_class):
    destination = os.path.join(output_base, chosen_class, os.path.basename(video_path))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(video_path, destination)
    
    if current_class and current_class != "N/A" and current_class != chosen_class:
        old_dest = os.path.join(output_base, current_class, os.path.basename(video_path))
        if os.path.exists(old_dest):
            os.remove(old_dest)
    return destination

def find_matching_image(video_name):
    base_name = os.path.splitext(video_name)[0]
    image_extensions = [".jpg", ".jpeg", ".png"]
    for folder in [images_folder, clips_folder]:
        for ext in image_extensions:
            candidate = os.path.join(folder, base_name + ext)
            if os.path.exists(candidate):
                return candidate
    return None

def overlay_text(image, lines, start_y=30, line_height=30, color=(0, 255, 0)):
    for i, line in enumerate(lines):
        y = start_y + i * line_height
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# We map the clips by sorting the DB rows chronologically per original video file.
# Since split_video generates clip_000.mp4, clip_001.mp4, etc., we can link them.
# We will pull all records from the DB to map them.
cursor = db._conn.execute("SELECT * FROM measurements ORDER BY timestamp ASC")
all_records = [dict(row) for row in cursor.fetchall()]

# Create a mapping of video_file -> list of rows
records_by_video = {}
for r in all_records:
    vid = r['video_file']
    if vid not in records_by_video:
        records_by_video[vid] = []
    records_by_video[vid].append(r)

for video_name in videos:
    # Example parsing: "clip_2026-04-22_16-05-01_Ethanol_005.mp4" 
    # OR if your split_video just uses "clip_0_5.mp4" 
    # For now, we will extract the original video and clip index from however your split_video names them.
    # Let's assume you've structured it so you can find the correct row:
    
    # --- TEMPORARY MAPPING LOGIC (Adjust based on how split_video names clips) ---
    # Assuming video_name contains the row ID (e.g., clip_ID.mp4) or we iterate through sequentially
    # Let's do a sequential grab if you are doing it folder by folder
    
    import re
    match = re.match(r"clip_(\d+)_(\d+)\.mp4", video_name)
    if not match:
        print(f"Skipping (unexpected filename format): {video_name}")
        continue
    
    experiment_idx = int(match.group(1))
    sample_idx = int(match.group(2))
    
    # Map to DB: Fetch all distinct main videos from DB to figure out which one is "experiment_idx"
    unique_main_videos = list(records_by_video.keys())
    if experiment_idx >= len(unique_main_videos):
        print(f"Skipping {video_name}: experiment_idx {experiment_idx} exceeds DB sessions.")
        continue
        
    main_video_name = unique_main_videos[experiment_idx]
    session_rows = records_by_video[main_video_name]
    
    if sample_idx >= len(session_rows):
        print(f"Skipping {video_name}: sample_idx {sample_idx} exceeds DB rows for session.")
        continue
        
    # GET THE ACTUAL DATABASE ROW
    db_row = session_rows[sample_idx]
    db_id = db_row['id']
    
    current_manual_class = db_row.get("manual_classification", "N/A")
    voltage = db_row.get("target_voltage", "N/A")
    flow_rate = db_row.get("flow_rate", "N/A")

    video_path = os.path.join(clips_folder, video_name)
    cap = cv2.VideoCapture(video_path)

    image_path = find_matching_image(video_name)
    image = cv2.imread(image_path) if image_path else None

    print(f"\nLabeling: {video_name}  |  DB Row ID: {db_id}")
    print(f"  Current Manual Class: {current_manual_class}")
    print(f"  Voltage: {voltage}  |  Flow Rate: {flow_rate}")

    if image is not None:
        image_overlay = image.copy()
        overlay_text(image_overlay, [
            f"Class: {current_manual_class}",
            f"V: {voltage}",
            f"Q: {flow_rate}",
        ])
        cv2.imshow("Image", image_overlay)

    labeled = False
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_overlay = frame.copy()
        overlay_text(frame_overlay, [
            f"Class: {current_manual_class}",
            f"V: {voltage}",
            f"Q: {flow_rate}",
        ])
        cv2.imshow("Video", frame_overlay)

        key = cv2.waitKey(33) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            db.close()
            print("Quit — Database connection closed.")
            exit()

        elif key == ord('n'):
            print("  Skipped")
            break

        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            class_index = int(chr(key)) - 1
            chosen_class = classes[class_index]

            # Write directly to the SQLite Database
            db.update_manual_classification(db_id, chosen_class)

            ensure_classified_copy(video_path, chosen_class, current_manual_class)
            cap.release()
            cv2.destroyAllWindows()
            print(f"  Labeled as '{chosen_class}' → DB updated.")
            labeled = True
            break

    if not labeled:
        cap.release()
        cv2.destroyAllWindows()

db.close()
print("\nDone labeling. Database connection closed.")