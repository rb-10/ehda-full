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

# your imports
from image_classification.integrated_pipeline.split_video import split_video
from image_classification.pre_processing_ben import *
from image_classification.integrated_pipeline.classify_images import classify_images
from image_classification.integrated_pipeline.update_jsons import update_jsons


# -------- SETTINGS --------#
save_electrospray = Path(r"data")
solution = "DMF"
folder = save_electrospray / solution

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

    for file_name in Path(folder / 'raw').glob('*.mp4'):
        output_folder = split_video(folder, file_name)
        all_chunks.extend(list(Path(output_folder).glob('*.mp4')))

    # -------- Process Videos (PARALLEL) --------#
    processed_clips = folder / 'PROCESSED CLIPS'
    os.makedirs(processed_clips, exist_ok=True)

    num_workers = multiprocessing.cpu_count() - 2

    tasks = [(vf, processed_clips) for vf in all_chunks]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_video, tasks))

    # -------- Classify --------#
    INPUT_FOLDER = processed_clips
    OUTPUT_BASE = folder / 'CLASSIFIED'
    JSON_FOLDER = folder / 'Current'

    results_csv = classify_images(
        model_path=MODEL_PATH,
        input_folder=INPUT_FOLDER,
        output_base=OUTPUT_BASE,
        confidence_threshold=0.70
    )

    stats = update_jsons(
        json_folder=JSON_FOLDER,
        results_csv=results_csv,
        reclassify_existing=True
    )

    print(stats)