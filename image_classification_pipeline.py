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
import cv2
import pandas as pd

# Update imports
from image_classification.integrated_pipeline.split_video import split_video
from image_classification.pre_processing_ben import *
from image_classification.integrated_pipeline.classify_images import classify_images
from mapping.software.database import ElectrosprayDatabase # Import your DB class


# -------- SETTINGS --------#
images_data_folder = Path(r"data/images")
master_data_folder = Path(r"data")
RAW_VIDEO_DIR = images_data_folder / "raw"
SLPIT_VIDEO_DIR = images_data_folder / "SPLIT"
# Where the script will dump the output
PROCESSED_CLIPS_DIR = images_data_folder / 'PROCESSED CLIPS'
CLASSIFIED_DIR = images_data_folder / 'CLASSIFIED'

MODEL_PATH = "image_classification/final_model/export.pkl"


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


if __name__ == "__main__":

    # -------- Split Videos --------#
    all_chunks = []

    for file_name in RAW_VIDEO_DIR.glob('*.mp4'):
        output_folder = split_video(SLPIT_VIDEO_DIR, file_name)
        all_chunks.extend(list(Path(output_folder).glob('*.mp4')))

    # -------- Process Videos (PARALLEL) --------#

    os.makedirs(PROCESSED_CLIPS_DIR, exist_ok=True)
    cpu_count = multiprocessing.cpu_count()
    num_workers = cpu_count // 2

    tasks = [(vf, PROCESSED_CLIPS_DIR) for vf in all_chunks]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_video, tasks, chunksize = 4))

    # -------- Classify --------#
    INPUT_FOLDER = PROCESSED_CLIPS_DIR
    OUTPUT_BASE = CLASSIFIED_DIR

    # This creates a CSV mapping clip names to predictions
    results_csv = classify_images(
        model_path=MODEL_PATH,
        input_folder=INPUT_FOLDER,
        output_base=OUTPUT_BASE,
        confidence_threshold=0.70
    )

    # -------- UPDATE DATABASE -------- #
    print("Updating Database with image classifications...")
    db = ElectrosprayDatabase(str(master_data_folder)) # Point to parent folder where data.db lives
    
    # Read the results CSV
    df = pd.read_csv(results_csv)
    
    # Assuming your clips are named something like clip_0_5.mp4 (experiment_idx, sample_idx)
    # We map that sample_idx to the correct row in the database.
    # We find all unique original videos to group by
    unique_videos = df['original_video'].unique() if 'original_video' in df.columns else []

    updated_count = 0
    for main_video in unique_videos:
        # Fetch ordered rows from DB that correspond to this main video
        db_rows = db.get_measurements_by_video(main_video)
        
        # Filter CSV for clips belonging to this main video
        video_clips = df[df['original_video'] == main_video].sort_values('clip_filename')
        
        # Link the clips to the DB rows chronologically
        for i, (_, row_data) in enumerate(video_clips.iterrows()):
            if i < len(db_rows):
                db_id = db_rows[i]['id']
                pred_class = row_data['predicted_class']
                db.update_image_classification(db_id, pred_class)
                updated_count += 1
            else:
                print(f"[WARNING] More clips generated than database rows for {main_video}")

    db.close()
    print(f"Database update complete. {updated_count} rows classified.")