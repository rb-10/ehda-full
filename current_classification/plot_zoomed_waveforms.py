"""
plot_zoomed_waveforms.py
------------------------
Plots raw oscilloscope waveforms for randomly selected samples, showing both
the full waveform and a zoomed-in view of the middle 10% side-by-side.

HOW TO USE:
  1. Optionally tweak N_SAMPLES and RANDOM_SEED for the layout.
  2. Run from the project root:  python current_classification/plot_zoomed_waveforms.py
  3. The script will use the interactive database prompt to let you select the solution(s).
"""

import sys
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ── Add project root to path ──────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mapping.software.database import ElectrosprayDatabase
from current_classification.train_code import SAMPLING_FREQ, INVALID_LABELS, CUTOFF_HZ, MULTIPLIER_NA

# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ── edit these values
# ══════════════════════════════════════════════════════════════════════════════

N_SAMPLES   = 2       # total number of random samples to draw (fewer is better since it's 2 plots per sample)
RANDOM_SEED = 42      # set to None for a different pick each run

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

def plot_waveforms(df_samples: pd.DataFrame, raw_dir: Path, b, a, output_path: Path, solutions_str: str):
    # Sort by label so they are visually grouped together
    df_samples = df_samples.sort_values(by="manual_classification").reset_index(drop=True)
    
    n_total = len(df_samples)
    
    # Create figure: n_total rows, 2 columns (Left: Full, Right: Zoomed)
    # sharex='col' shares the X axis within columns (so all full plots share X, all zoomed plots share X)
    # sharey=True shares the Y axis across the entire grid
    fig, axes = plt.subplots(n_total, 2, figsize=(14, n_total * 3.5), squeeze=False, sharex='col', sharey=True)
    fig.suptitle(
        f"Waveform Comparison (Full vs Middle 10% Zoom) — Solutions: {solutions_str}\n"
        f"({n_total} random samples, seed={RANDOM_SEED})",
        fontsize=14, fontweight="bold", y=1.02
    )

    for idx, (_, sample_row) in enumerate(df_samples.iterrows()):
        ax_full = axes[idx, 0]
        ax_zoom = axes[idx, 1]
        
        label = sample_row["manual_classification"]
        colour = LABEL_COLOURS.get(label, DEFAULT_COLOUR)
        file_path = raw_dir / str(sample_row["raw_data_file"])
        
        if not file_path.exists():
            for ax in [ax_full, ax_zoom]:
                ax.text(0.5, 0.5, "file not found", ha="center", va="center", color="red", fontsize=10)
                ax.set_title(f"Label: {label} | ID: {sample_row.get('id', '?')}", fontsize=10)
            continue
            
        try:
            raw, filtered = load_and_filter_waveform(file_path, b, a)
            t_ms = build_time_axis(len(raw), SAMPLING_FREQ)
            
            # Plot on both axes
            for ax in [ax_full, ax_zoom]:
                ax.plot(t_ms, raw, color=colour, alpha=0.35, linewidth=0.8, label="Raw")
                if SHOW_FILTERED:
                    ax.plot(t_ms, filtered, color=colour, alpha=0.9, linewidth=1.5, label="Filtered")
                
                ax.grid(True, linestyle="--", alpha=0.5)
            
            # Formatting Full plot
            ax_full.set_title(f"FULL: {label} | Sol: {sample_row.get('solution_name', '?')} | ID: {sample_row.get('id', '?')}", fontsize=10)
            ax_full.set_ylabel("Current (nA)", fontsize=9)
            ax_full.set_ylim(-40, 140)
            if idx == n_total - 1:
                ax_full.set_xlabel("Time (ms)", fontsize=9)
            if idx == 0 and SHOW_FILTERED:
                ax_full.legend(fontsize=8, loc="upper right")
                
            # Formatting Zoomed plot (middle 10%)
            total_time_ms = t_ms[-1]
            start_zoom = 0.45 * total_time_ms
            end_zoom   = 0.55 * total_time_ms
            ax_zoom.set_xlim(start_zoom, end_zoom)
            ax_zoom.set_ylim(-40, 140)
            
            ax_zoom.set_title(f"ZOOMED (Middle 10%): {label} | ID: {sample_row.get('id', '?')}", fontsize=10)
            if idx == n_total - 1:
                ax_zoom.set_xlabel("Time (ms)", fontsize=9)
                
        except Exception as exc:
            for ax in [ax_full, ax_zoom]:
                ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center", color="red", fontsize=9, wrap=True)
                ax.set_title(f"Label: {label} | ID: {sample_row.get('id', '?')}", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path.resolve()}")


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

    # ── 2. Filter out invalid labels ──────────────────────────────────────────
    df_valid = df_db[
        df_db["manual_classification"].notna() &
        (~df_db["manual_classification"].isin(INVALID_LABELS))
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

    label_counts = df_sampled["manual_classification"].value_counts().to_dict()
    print(f"[3/4] Selected {n} samples  ->  label breakdown: {label_counts}")

    # ── 4. Build filter & plot ─────────────────────────────────────────────────
    b, a = _make_butter_filter(CUTOFF_HZ, SAMPLING_FREQ)

    # Safe filename if multiple solutions
    if len(solutions_selected) == 1:
        fname = f"{solutions_selected[0]}_zoomed_waveforms.png"
    else:
        fname = "MultiSolution_zoomed_waveforms.png"
        
    output_path = OUTPUT_DIR / fname

    print(f"[4/4] Generating zoomed waveform plot …")
    plot_waveforms(df_sampled, RAW_DIR, b, a, output_path, solutions_str)

    print("\nDone.")


if __name__ == "__main__":
    main()
