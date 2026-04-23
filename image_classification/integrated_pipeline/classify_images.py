"""
classify_images.py

ML-only script:
- Runs inference
- Copies images to class folders
- Saves results to CSV

Can be used standalone OR imported as a function.
"""

import os
import re
import shutil
from pathlib import Path
import pandas as pd
from fastai.vision.all import load_learner, PILImage
from datetime import datetime


CLASSES = ["cone_jet", "dripping", "intermitent", "multi_jet", "unconclusive"]


def classify_images(
    model_path: str,
    input_folder: Path,
    output_base: Path,
    confidence_threshold: float = 0.80,
    run_id: str | None = None,
) -> Path:
    """
    Runs classification and returns path to results CSV.
    Compatible with timestamped filenames: clip_YYYY-MM-DD_HH-MM-SS_SOL_IDX.jpg
    """

    input_folder = Path(input_folder)
    output_base = Path(output_base)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_base / f"classification_results_{run_id}.csv"

    # ── Setup ─────────────────────────────────────────────
    for cls in CLASSES:
        os.makedirs(output_base / cls, exist_ok=True)

    print(f"[ML] Loading model: {model_path}")
    learn = load_learner(model_path)

    images = sorted(input_folder.glob("*.jpg")) + \
             sorted(input_folder.glob("*.png"))

    if not images:
        print("[ML] No images found.")
        return results_path

    results = []
    print(f"[ML] Found {len(images)} images\n")

    for img_path in images:
        # NEW FILENAME LOGIC:
        # clip_2026-04-23_12-12-47_EW82_000.jpg
        # 1. Remove 'clip_' prefix
        # 2. Split at the LAST underscore to separate the Index from the Video Name
        filename = img_path.name
        
        try:
            # clip_2026-04-23_12-12-47_EW82_000.jpg -> 2026-04-23_12-12-47_EW82_000.jpg
            clean_name = filename.replace("clip_", "") 
            
            # rsplit separates from the right: ['2026-04-23_12-12-47_EW82', '000.jpg']
            base_video_part, index_part = clean_name.rsplit('_', 1)
            
            original_video = base_video_part + ".mp4"
            sample_idx = int(index_part.split('.')[0]) # Extract 000 from 000.jpg
        except Exception:
            print(f"[SKIP] Filename format mismatch: {filename}")
            continue

        img = PILImage.create(img_path)
        label, _, probs = learn.predict(img)

        raw_class  = str(label)
        confidence = probs.max().item()
        predicted_class = raw_class.lower().replace(" ", "_")

        final_class = predicted_class if confidence >= confidence_threshold else "unconclusive"

        if final_class not in CLASSES:
            final_class = "unconclusive"

        # Copy image to classified folder
        dest = output_base / final_class / img_path.name
        shutil.copy2(img_path, dest)

        results.append({
            "clip_filename": filename,
            "original_video": original_video,  # Key for DB linking
            "sample_idx": sample_idx,
            "raw_class": raw_class,
            "predicted_class": f"{raw_class} ({confidence:.0%})",
            "final_class": final_class,
            "confidence": confidence,
            "run_id": run_id
        })

        print(f"{filename:<45} {final_class:<15} {confidence:.2%}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(results_path, index=False)
        print(f"\n[ML] Results saved to: {results_path}")
    else:
        print("\n[ML] No results to save.")
        
    return results_path

# ── Standalone usage ────────────────────────────────────────
def main():
    MODEL_PATH = "final_model/export.pkl"
    SOLUTION = "DMF"
    INPUT_FOLDER = Path(rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\{SOLUTION}\PROCESSED CLIPS")
    OUTPUT_BASE  = Path(rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\{SOLUTION}\CLASSIFIED")

    classify_images(
        model_path=MODEL_PATH,
        input_folder=INPUT_FOLDER,
        output_base=OUTPUT_BASE,
        confidence_threshold=0.80,
    )


if __name__ == "__main__":
    main()