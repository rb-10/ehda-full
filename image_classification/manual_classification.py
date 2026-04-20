import cv2
import os
import shutil
import json
import re

liquid = "Ethanol" 
base = r"C:\Users\Bruno Duarte\Documents\datasets"
json_folder = os.path.join(base, liquid, "Current")
images_folder = os.path.join(base, liquid, "PROCESSED CLIPS")
clips_folder = os.path.join(base, liquid, "SPLIT CLIPS")
output_base = os.path.join(base, liquid, "CLASSIFIED")

# If json_files is empty, all JSON files in json_folder will be loaded.
# Otherwise put the exact JSON filenames you want to use here.
json_files = []

classes = ["cone_jet", "dripping", "intermitent", "multi_jet", "unconclusive", "undefined"]

for cls in classes:
    os.makedirs(os.path.join(output_base, cls), exist_ok=True)

videos = [f for f in os.listdir(images_folder) if f.endswith(".mp4")]
videos.sort()

print("Controls:")
print("1–6 → assign class")
print("q   → quit")
print("n   → skip video")

# Cache loaded JSON files to avoid reloading repeatedly
json_cache = {}

def load_experiment_json(experiment_idx):
    """Load and cache the JSON for a given experiment index."""
    if experiment_idx in json_cache:
        return json_cache[experiment_idx]
    
    json_filename = f"experiment_{experiment_idx}.json"
    json_path = os.path.join(json_folder, json_filename)
    
    if not os.path.exists(json_path):
        print(f"  [WARNING] JSON not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    json_cache[experiment_idx] = (json_path, data)
    return json_cache[experiment_idx]

def save_experiment_json(experiment_idx):
    """Save the cached JSON back to disk."""
    if experiment_idx not in json_cache:
        return
    json_path, data = json_cache[experiment_idx]
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Saved JSON: {os.path.basename(json_path)}")


def get_json_file_list():
    if json_files:
        selected = [f for f in json_files if os.path.exists(os.path.join(json_folder, f))]
        missing = [f for f in json_files if not os.path.exists(os.path.join(json_folder, f))]
        if missing:
            print(f"  [WARNING] Missing JSON files: {missing}")
        return selected
    return [f for f in os.listdir(json_folder) if f.endswith('.json')]

json_list = get_json_file_list()
print(f"Using JSON files: {json_list}")


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

    for folder in [images_folder, clips_folder]:
        for entry in os.listdir(folder):
            name, ext = os.path.splitext(entry)
            if ext.lower() in image_extensions and base_name in name:
                return os.path.join(folder, entry)

    return None


def overlay_text(image, lines, start_y=30, line_height=30, color=(0, 255, 0)):
    for i, line in enumerate(lines):
        y = start_y + i * line_height
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

for video_name in videos:
    # Parse experiment and sample index from filename: clip_yyy_xxx.mp4
    match = re.match(r"clip_(\d+)_(\d+)\.mp4", video_name)
    if not match:
        print(f"Skipping (unexpected filename format): {video_name}")
        continue

    experiment_idx = int(match.group(1))
    sample_idx = int(match.group(2))
    sample_key = f"sample {sample_idx}"

    # Load the corresponding JSON
    result = load_experiment_json(experiment_idx)
    if result is None:
        print(f"  Skipping {video_name} — no matching JSON for experiment {experiment_idx}")
        continue

    json_path, data = result

    if sample_key not in data:
        print(f"  [WARNING] '{sample_key}' not found in {os.path.basename(json_path)}, skipping.")
        continue

    sample_data = data[sample_key]
    image_classification = sample_data.get("image_classification", "N/A")
    voltage = sample_data.get("voltage", "N/A")
    flow_rate = sample_data.get("flow_rate", "N/A")

    video_path = os.path.join(images_folder, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening:", video_name)
        continue

    image_path = find_matching_image(video_name)
    image = None
    if image_path is not None:
        image = cv2.imread(image_path)
        if image is None:
            print(f"  [WARNING] Failed to load image: {image_path}")
            image_path = None

    print(f"\nLabeling: {video_name}  |  Experiment: {experiment_idx}, Sample: {sample_idx}")
    print(f"  Current image_classification: {image_classification}")
    print(f"  voltage: {voltage}  |  flow_rate: {flow_rate}")
    if image_path:
        print(f"  Showing image: {os.path.basename(image_path)}")
    else:
        print("  No matching image found; only showing video.")

    if image is not None:
        image_overlay = image.copy()
        overlay_text(image_overlay, [
            f"image_classification: {image_classification}",
            f"voltage: {voltage}",
            f"flow_rate: {flow_rate}",
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
            f"image_classification: {image_classification}",
            f"voltage: {voltage}",
            f"flow_rate: {flow_rate}",
        ])
        cv2.imshow("Video", frame_overlay)

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 33
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            # Save all modified JSONs before quitting
            for exp_idx in list(json_cache.keys()):
                save_experiment_json(exp_idx)
            cap.release()
            cv2.destroyAllWindows()
            print("Quit — all changes saved.")
            exit()

        elif key == ord('n'):
            print("  Skipped")
            break

        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            class_index = int(chr(key)) - 1
            chosen_class = classes[class_index]

            sample_data["image_classification"] = chosen_class
            save_experiment_json(experiment_idx)

            ensure_classified_copy(video_path, chosen_class, image_classification)
            cap.release()
            cv2.destroyAllWindows()
            print(f"  Labeled as '{chosen_class}' → copied to CLASSIFIED/{chosen_class}/")
            labeled = True
            break

    if not labeled:
        cap.release()
        cv2.destroyAllWindows()

print("\nDone labeling.")
# Final save of any remaining cached JSONs
for exp_idx in list(json_cache.keys()):
    save_experiment_json(exp_idx)