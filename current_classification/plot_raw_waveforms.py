"""
plot_raw_waveforms.py
---------------------
Plots raw oscilloscope waveforms for randomly selected samples, 
sorted by spray mode (manual_classification label).

HOW TO USE:
  1. Optionally tweak N_SAMPLES, RANDOM_SEED, and N_COLS for the layout.
  2. Run from the project root:  python current_classification/plot_raw_waveforms.py
  3. The script will use the interactive database prompt to let you select the solution(s).
"""

import sys
import os
import random
import math
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ── Add project root to path ──────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mapping.software.database import ElectrosprayDatabase
from current_classification.train_code import INVALID_LABELS, CUTOFF_HZ, MULTIPLIER_NA

# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ── edit these values
# ══════════════════════════════════════════════════════════════════════════════

N_SAMPLES   = 8     # total number of random samples to draw
RANDOM_SEED = 40      # set to None for a different pick each run

# Max columns for the plot grid (2 or 3 is best for visibility)
N_COLS = 2

# Where plots are saved (relative to project root)
OUTPUT_DIR = Path("current_classification/plots/raw_waveforms")

# Whether to also overlay the low-pass filtered signal on each subplot
SHOW_FILTERED = True

# ══════════════════════════════════════════════════════════════════════════════

# ── Colour palette, one colour per spray mode ─────────────────────────────────
LABEL_COLOURS = {
    'dripping': "#2e62d4",
    'intermitent': "#066400",
    'cone_jet': "#e80101",
    'multi_jet': "#830068",
}
DEFAULT_COLOUR = "#9E9E9E"

def _make_butter_filter(cutoff_hz: float, sample_rate: float):
    cutoff_norm = cutoff_hz / (0.5 * sample_rate)
    b, a = butter(6, Wn=cutoff_norm, btype="low", analog=False)
    return b, a

def load_and_filter_waveform(file_path: Path, b, a) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(file_path) * MULTIPLIER_NA
    filtered = filtfilt(b, a, raw)
    return raw, filtered

def build_time_axis(n_points: int, sample_rate: float) -> np.ndarray:
    return np.arange(n_points) / sample_rate * 1_000   # ms

def plot_waveforms(df_samples: pd.DataFrame, raw_dir: Path, output_path: Path, solutions_str: str):
    # Sort by label so they are visually grouped together
    df_samples = df_samples.sort_values(by="final_label").reset_index(drop=True)
    
    n_total = len(df_samples)
    n_rows = math.ceil(n_total / N_COLS)
    
    # Create figure with a better layout (not too wide)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 7, n_rows * 3.5), squeeze=False, sharex=True, sharey=True)
    fig.suptitle(
        f"Raw oscilloscope readings — Solutions: {solutions_str}\n"
        f"({n_total} random samples, seed={RANDOM_SEED})",
        fontsize=14, fontweight="bold", y=1.02
    )

    for idx, (_, sample_row) in enumerate(df_samples.iterrows()):
        r = idx // N_COLS
        c = idx % N_COLS
        ax = axes[r, c]
        
        label = sample_row["final_label"]
        colour = LABEL_COLOURS.get(label, DEFAULT_COLOUR)
        file_path = raw_dir / str(sample_row["raw_data_file"])
        
        if not file_path.exists():
            ax.text(0.5, 0.5, "file not found", ha="center", va="center", color="red", fontsize=10)
            ax.set_title(f"Label: {label} | ID: {sample_row.get('id', '?')} | V: {sample_row.get('target_voltage', '?')} | Flow: {sample_row.get('flow_rate', '?')}", fontsize=10)
            continue
            
        try:
            sample_rate = float(sample_row.get("sample_rate", 100000.0))
            if sample_rate < 6000:
                raw = np.load(file_path) * MULTIPLIER_NA
                filtered = raw
                show_filtered_for_this = False
            else:
                b, a = _make_butter_filter(CUTOFF_HZ, sample_rate)
                raw, filtered = load_and_filter_waveform(file_path, b, a)
                show_filtered_for_this = SHOW_FILTERED

            t_ms = build_time_axis(len(raw), sample_rate)
            
            # Plot raw
            # Plot filtered
            if show_filtered_for_this:
                ax.plot(t_ms, raw, color=colour, alpha=0.35, linewidth=0.8, label="Raw")
                ax.plot(t_ms, filtered, color=colour, alpha=0.9, linewidth=1.5, label="Filtered")
            else:
                ax.plot(t_ms, raw, color=colour, alpha=0.9, linewidth=0.8, label="Raw")
            
            
            ax.set_title(f"Label: {label} | ID: {sample_row.get('id', '?')} | V: {sample_row.get('target_voltage', '?')} | Flow: {sample_row.get('flow_rate', '?')}", fontsize=10)
            ax.set_xlabel("Time (ms)", fontsize=9)
            ax.set_ylabel("Current (nA)", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            
            if idx == 0 and show_filtered_for_this:
                ax.legend(fontsize=8, loc="upper right")
                
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center", color="red", fontsize=9, wrap=True)
            ax.set_title(f"Label: {label} | ID: {sample_row.get('id', '?')}", fontsize=10)

    # Hide any unused subplots
    for idx in range(n_total, n_rows * N_COLS):
        r = idx // N_COLS
        c = idx % N_COLS
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path.resolve()}")

def clean_label(label):
    """
    Strips confidence annotations like '(98%)' from image_classification labels,
    normalizes whitespace/case, and returns None for empty/invalid values.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    label = str(label)
    # Remove anything in parentheses, e.g. "cone_jet (98%)" -> "cone_jet"
    label = re.sub(r"\(.*?\)", "", label).strip()
    return label if label else None

def resolve_label(row):
    """
    Returns manual_classification if it's valid, otherwise falls back to
    a cleaned-up image_classification label.
    """
    manual = row.get("manual_classification")
    if pd.notna(manual) and manual not in INVALID_LABELS:
        return manual

    fallback = clean_label(row.get("image_classification"))
    return fallback  # could still be None/invalid, filtered out later

def main():
    BASE    = project_root / "data"
    RAW_DIR = BASE / "raw_waveforms"

    print(f"[1/4] Loading database from: {BASE}")
    db = ElectrosprayDatabase(str(BASE))
    
    # db.load_training_dataframe() will interactively ask for solution(s)
    df_db = db.load_training_dataframe()
    
    if df_db.empty:
        raise ValueError("No data returned from database.")
    
    # Get the names of the solutions that were selected
    SOLUTION_COL = "solution_name"
    solutions_selected = []
    if SOLUTION_COL in df_db.columns:
        solutions_selected = df_db[SOLUTION_COL].dropna().unique().tolist()
        solutions_str = ", ".join(solutions_selected)
    else:
        solutions_str = "Unknown"

    print(f"[2/4] Found {len(df_db)} samples across selected solutions: {solutions_str}")
    
    if "sample_rate" not in df_db.columns:
        df_db["sample_rate"] = 100000.0
    df_db["sample_rate"] = df_db["sample_rate"].fillna(100000.0)
    
    df_db["final_label"] = df_db.apply(resolve_label, axis=1)
    # ── 2. Filter out invalid labels ──────────────────────────────────────────
    df_valid = df_db[
        df_db["final_label"].notna() &
        (~df_db["final_label"].isin(INVALID_LABELS))
    ].copy()

    if df_valid.empty:
        raise ValueError("All samples for the selected solutions have invalid/missing labels.")

    print(f"       {len(df_valid)} samples have valid labels.")

    # ── 3. Random sample of N_SAMPLES ─────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    n   = min(N_SAMPLES, len(df_valid))

    if n < N_SAMPLES:
        print(f"  ⚠  Only {n} valid samples available (requested {N_SAMPLES}).")

    sampled_indices = rng.sample(list(df_valid.index), n)
    df_sampled = df_valid.loc[sampled_indices].copy()

    label_counts = df_sampled["final_label"].value_counts().to_dict()
    print(f"[3/4] Selected {n} samples  ->  label breakdown: {label_counts}")

    # Safe filename if multiple solutions
    if len(solutions_selected) == 1:
        fname = f"{solutions_selected[0]}_raw_waveforms.png"
    else:
        fname = "MultiSolution_raw_waveforms.png"
        
    output_path = OUTPUT_DIR / fname

    print(f"[4/4] Generating waveform plot …")
    plot_waveforms(df_sampled, RAW_DIR, output_path, solutions_str)

    print("\nDone.")


if __name__ == "__main__":
    main()