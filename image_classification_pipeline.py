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

    # Load ALL measurements once and pre-group by video_file for fast lookup.
    # Rows within each video are sorted by id (insertion order = experiment order).
    all_rows = pd.read_sql("SELECT * FROM measurements", db._conn)
    rows_by_video = {
        video: grp.sort_values('id').reset_index(drop=True)
        for video, grp in all_rows.groupby('video_file')
    }

    # Diagnostic summary so mismatches are easy to spot
    print("Video → DB row counts:")
    for v, grp in rows_by_video.items():
        print(f"  {v}: {len(grp)} rows ({len(grp) // SAMPLES_PER_STEP} steps)")

    updated_count = 0
    skipped_count = 0
    warn_count    = 0

    for _, clip_row in df.iterrows():
        clip_filename = clip_row['clip_filename']
        pred_class    = clip_row['predicted_class']

        original_name, experiment_idx = parse_clip_name(clip_filename)
        if original_name is None:
            print(f"  [WARNING] Could not parse clip name: {clip_filename} — skipping")
            warn_count += 1
            continue

        # Match original_name against the video_file column.
        # video_file may include an extension (.mp4) so we try both.
        video_key = next(
            (k for k in rows_by_video
             if Path(k).stem == original_name or k == original_name),
            None
        )
        if video_key is None:
            print(f"  [WARNING] '{original_name}' not found in video_file column — skipping {clip_filename}")
            print(f"            Available video_file values: {list(rows_by_video.keys())}")
            warn_count += 1
            continue

        video_rows = rows_by_video[video_key]

        # experiment_idx resets to 0 per video, so index directly into that video's rows
        start = experiment_idx * SAMPLES_PER_STEP
        end   = start + SAMPLES_PER_STEP
        target_rows = video_rows.iloc[start:end]

        if target_rows.empty:
            print(f"  [WARNING] Experiment index {experiment_idx} out of range for "
                  f"'{video_key}' ({len(video_rows)} rows) — skipping {clip_filename}")
            warn_count += 1
            continue

        # Skip if every row in this block is already classified
        already_classified = (
            target_rows['image_classification'].notna() &
            (target_rows['image_classification'].str.strip() != '') &
            (target_rows['image_classification'].str.strip() != 'N/A')
        )
        if already_classified.all():
            skipped_count += 1
            continue

        # Write the same classification to all SAMPLES_PER_STEP rows
        for db_id in target_rows['id']:
            db.update_image_classification(db_id, pred_class)
            updated_count += 1

        print(f"  [OK] {clip_filename} → '{pred_class}' "
              f"(video '{video_key}', step {experiment_idx}, {len(target_rows)} rows)")

    db.close()
    print(f"\nDone. {updated_count} rows updated, "
          f"{skipped_count} steps skipped (already classified), "
          f"{warn_count} warnings.")