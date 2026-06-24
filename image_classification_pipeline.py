"""
This script processes raw videos in three stages:
1. Segmentation: Splits raw video into 40-frame clips.
2. Analysis:     Processes and classifies each clip.
3. Integration:  Links results to the corresponding experiment JSON.

PREREQUISITES:
    pip install fastai pandas IPython

FILE STRUCTURE REQUIREMENTS:
    Your main directory (defined by 'save_electrospray') should look like this:

    [Solution Folder]           <-- Set solution name in script
    ├── raw/                    <-- Video files
    │   ├── 000.mp4             <-- Index must match JSON
    │   ├── 001.mp4
    │   └── ...
    └── Current/                <-- Metadata files
        ├── experiment_0.json   <-- Matches 000.mp4
        ├── experiment_1.json   <-- Matches 001.mp4
        └── ...

INSTRUCTIONS:
    1. Update 'save_electrospray' to your main parent folder path.
    2. Update the solution subfolder name as needed.
    3. Ensure videos are .mp4 and JSONs are named 'experiment_X.json'.


"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import re
import cv2
import pandas as pd

# Update imports
from image_classification.integrated_pipeline.split_video import split_video
from image_classification.pre_processing_ben import *
from image_classification.integrated_pipeline.classify_images import classify_images
from mapping.software.database import ElectrosprayDatabase


# -------- SETTINGS --------#
images_data_folder = Path(r"data/images")
master_data_folder = Path(r"data")
RAW_VIDEO_DIR      = images_data_folder / "raw"
SLPIT_VIDEO_DIR    = images_data_folder / "SPLIT"
PROCESSED_CLIPS_DIR = images_data_folder / 'PROCESSED CLIPS'
CLASSIFIED_DIR     = images_data_folder / 'CLASSIFIED'

MODEL_PATH = "image_classification/final_model/export.pkl"

# How many DB rows exist per voltage/flow-rate step
SAMPLES_PER_STEP = 5


def process_video(args):
    video_file, processed_clips = args

    out_img_path = processed_clips / (video_file.stem + '.png')

    if out_img_path.exists():
        print(f"Skipping {video_file}")
        return

    cap = cv2.VideoCapture(str(video_file))
    frames = read_gray_frames(cap)

    if not frames:
        return

    merged_image = cv2.merge((
        temporal_median_background(frames),
        tiny_particle_detector(frames),
        original_optical_flow(frames)
    ))

    processed_img = max_pool_to_size(merged_image, (256, 256))
    cv2.imwrite(str(out_img_path), processed_img)


def parse_clip_name(clip_filename):
    """
    Extract (original_video_name, experiment_index) from a clip filename.

    Expected format:  clip_{original_name}_{index}[.ext]
    Example:          clip_2026-06-24_10-58-40_TEST1_000.mp4
      → original_name  = '2026-06-24_10-58-40_TEST1'
        experiment_idx = 0
    """
    stem = Path(clip_filename).stem          # drop extension
    # Strip leading 'clip_'
    without_prefix = re.sub(r'^clip_', '', stem)
    # The experiment index is the last '_NNN' token (zero-padded digits)
    match = re.match(r'^(.+)_(\d+)$', without_prefix)
    if not match:
        return None, None
    original_name = match.group(1)
    experiment_idx = int(match.group(2))
    return original_name, experiment_idx


if __name__ == "__main__":

    # -------- Split Videos --------#
    all_chunks = []

    for file_name in RAW_VIDEO_DIR.glob('*.mp4'):
        output_folder = split_video(SLPIT_VIDEO_DIR, file_name)
        all_chunks.extend(list(Path(output_folder).glob('*.mp4')))

    # -------- Process Videos (PARALLEL) --------#
    os.makedirs(PROCESSED_CLIPS_DIR, exist_ok=True)
    cpu_count   = multiprocessing.cpu_count()
    num_workers = cpu_count // 2

    tasks = [(vf, PROCESSED_CLIPS_DIR) for vf in all_chunks]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_video, tasks, chunksize=4))

    # -------- Classify --------#
    results_csv = classify_images(
        model_path=MODEL_PATH,
        input_folder=PROCESSED_CLIPS_DIR,
        output_base=CLASSIFIED_DIR,
        confidence_threshold=0.70
    )

    # -------- UPDATE DATABASE -------- #
    print("Updating database with image classifications …")
    db = ElectrosprayDatabase(str(master_data_folder))

    df = pd.read_csv(results_csv)

    # Load ALL measurements once; we'll work in-memory
    all_rows = pd.read_sql("SELECT * FROM measurements", db._conn)

    updated_count = 0
    skipped_count = 0

    for _, clip_row in df.iterrows():
        clip_filename = clip_row['clip_filename']
        pred_class    = clip_row['predicted_class']

        original_name, experiment_idx = parse_clip_name(clip_filename)
        if original_name is None:
            print(f"  [WARNING] Could not parse clip name: {clip_filename} — skipping")
            continue

        # The solution name is the last '_'-separated token of the original video name
        # e.g. '2026-06-24_10-58-40_TEST1' → 'TEST1'
        solution_name = original_name.rsplit('_', 1)[-1]

        # Filter to this solution's rows, ordered the same way the DB was built
        # (assumes rows are stored in experiment order; adjust sort key if needed)
        solution_rows = (
            all_rows[all_rows['solution_name'].str.strip() == solution_name]
            .sort_values('id')
            .reset_index(drop=True)
        )

        if solution_rows.empty:
            print(f"  [WARNING] No DB rows found for solution '{solution_name}' — skipping {clip_filename}")
            continue

        # Each experiment index maps to a block of SAMPLES_PER_STEP consecutive rows
        start = experiment_idx * SAMPLES_PER_STEP
        end   = start + SAMPLES_PER_STEP
        target_rows = solution_rows.iloc[start:end]

        if target_rows.empty:
            print(f"  [WARNING] Experiment index {experiment_idx} out of range for '{solution_name}' — skipping")
            continue

        # Skip if every row in this block is already classified
        already_classified = target_rows['image_classification'].notna() & \
                             (target_rows['image_classification'].str.strip() != '') & \
                             (target_rows['image_classification'].str.strip() != 'N/A')

        if already_classified.all():
            print(f"  [SKIP] {clip_filename} — all {SAMPLES_PER_STEP} rows already classified")
            skipped_count += 1
            continue

        # Write the same classification to all SAMPLES_PER_STEP rows
        for db_id in target_rows['id']:
            db.update_image_classification(db_id, pred_class)
            updated_count += 1

        print(f"  [OK] {clip_filename} → '{pred_class}' written to {len(target_rows)} rows "
              f"(experiment {experiment_idx}, solution '{solution_name}')")

    db.close()
    print(f"\nDone. {updated_count} rows updated, {skipped_count} steps skipped (already classified).")