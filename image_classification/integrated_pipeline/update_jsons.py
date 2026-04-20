"""
update_jsons.py

Updates experiment JSONs based on classification CSV.

Can be used standalone OR imported.
"""

import json
from pathlib import Path
import pandas as pd


def update_jsons(
    json_folder: Path,
    results_csv: Path,
    reclassify_existing: bool = False,
    confidence_threshold: float = 0.80,
) -> dict:
    """
    Updates JSON files using classification results.

    Returns stats dict.
    """

    json_folder = Path(json_folder)
    results_csv = Path(results_csv)

    df = pd.read_csv(results_csv)

    print(f"[JSON] Loaded {len(df)} classification results")

    json_cache = {}

    def load_json(exp_idx):
        if exp_idx in json_cache:
            return json_cache[exp_idx]

        path = json_folder / f"experiment_{exp_idx}.json"
        if not path.exists():
            print(f"[WARN] Missing JSON: {path}")
            return None

        with open(path, "r") as f:
            data = json.load(f)

        json_cache[exp_idx] = [path, data]
        return json_cache[exp_idx]

    def save_json(exp_idx):
        path, data = json_cache[exp_idx]
        tmp_path = str(path) + ".tmp"

        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)

        Path(tmp_path).replace(path)

    updated = 0
    skipped = 0

    for _, row in df.iterrows():

        exp_idx = int(row["experiment_idx"])
        sample_idx = int(row["sample_idx"])
        sample_key = f"sample {sample_idx}"

        result = load_json(exp_idx)
        if result is None:
            skipped += 1
            continue

        path, data = result

        if sample_key not in data:
            print(f"[SKIP] {sample_key} missing in experiment_{exp_idx}")
            skipped += 1
            continue

        sample = data[sample_key]

        if "image_classification" in sample and not reclassify_existing:
            skipped += 1
            continue

        final_class = row["final_class"]
        confidence = row["confidence"]

        if confidence < confidence_threshold:
            final_class = "unconclusive"

        sample["image_classification"] = final_class
        updated += 1

    # Save once
    for exp_idx in json_cache:
        save_json(exp_idx)

    print(f"\n[JSON] Done: {updated} updated, {skipped} skipped")

    return {
        "updated": updated,
        "skipped": skipped,
        "total": len(df)
    }


# ── Standalone usage ────────────────────────────────────────
def main():
    SOLUTION = "DMF"
    JSON_FOLDER  = Path(rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\{SOLUTION}\Current")
    RESULTS_CSV = Path(rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\{SOLUTION}\CLASSIFIED\classification_results_1.csv")

    update_jsons(
        json_folder=JSON_FOLDER,
        results_csv=RESULTS_CSV,
        reclassify_existing=False,
        confidence_threshold=0.80,
    )


if __name__ == "__main__":
    main()