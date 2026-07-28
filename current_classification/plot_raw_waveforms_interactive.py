"""
plot_raw_waveforms_interactive.py
----------------------------------
Interactively browse raw oscilloscope waveforms, one at a time, in order
(sorted by id — no randomization). Use the LEFT and RIGHT arrow keys to
step backwards and forwards through the samples. Nothing is saved to
disk; everything is shown live on screen.

HOW TO USE:
  1. Optionally tweak N_SAMPLES (how many samples to load, or None for all).
  2. Run from the project root:  python current_classification/plot_raw_waveforms_interactive.py
  3. The script will use the interactive database prompt to let you select the solution(s).
  4. A window will open showing the first sample. Press:
       -> (right arrow)  : next sample
       <- (left arrow)   : previous sample
       q                 : quit / close the window
"""

import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
from scipy.signal import butter, filtfilt

# ── Force an interactive backend ───────────────────────────────────────────
# Something imported later (e.g. the database / train_code modules) may set
# a non-interactive backend like "Agg", which makes plt.show() return
# immediately without opening a window. Try a few common interactive
# backends here, before pyplot is imported, so the window actually appears.
_BACKENDS_TO_TRY = ["QtAgg", "Qt5Agg", "TkAgg", "MacOSX"]
_chosen_backend = None
for _bk in _BACKENDS_TO_TRY:
    try:
        matplotlib.use(_bk, force=True)
        _chosen_backend = _bk
        break
    except Exception:
        continue

import matplotlib.pyplot as plt

print(f"[backend] Using matplotlib backend: {matplotlib.get_backend()}"
      + ("" if _chosen_backend else "  (no interactive backend available — install PyQt5 or tkinter)"))

# ── Add project root to path ──────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mapping.software.database import ElectrosprayDatabase
from current_classification.train_code import INVALID_LABELS, CUTOFF_HZ, MULTIPLIER_NA

# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ── edit these values
# ══════════════════════════════════════════════════════════════════════════════

# How many samples to load for browsing. Set to None to load *all* valid
# samples for the selected solution(s).
N_SAMPLES = None

# Whether to also overlay the low-pass filtered signal on the plot
SHOW_FILTERED = True

# Fixed Y-axis limits (current, in nA) as a (min, max) tuple, e.g. (-50, 200).
# Set to None to let matplotlib auto-scale the Y-axis for each sample instead.
Y_LIMITS = (-100, 1000)

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


def load_and_filter_waveform(file_path: Path, cutoff_hz: float, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(file_path) * MULTIPLIER_NA
    b, a = _make_butter_filter(cutoff_hz, sample_rate)
    filtered = filtfilt(b, a, raw)
    return raw, filtered

def build_time_axis(n_points: int, sample_rate: float) -> np.ndarray:
    return np.arange(n_points) / sample_rate * 1_000  # ms


def clean_label(label):
    """
    Strips confidence annotations like '(98%)' from image_classification labels,
    normalizes whitespace/case, and returns None for empty/invalid values.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    label = str(label)
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


class WaveformBrowser:
    """Steps through a dataframe of samples, one waveform at a time,
    using the left/right arrow keys."""

    def __init__(self, df_samples: pd.DataFrame, raw_dir: Path, cutoff_hz: float, solutions_str: str):
        self.df = df_samples.reset_index(drop=True)
        self.raw_dir = raw_dir
        self.cutoff_hz = cutoff_hz
        self.solutions_str = solutions_str
        self.idx = 0
        self.n_total = len(self.df)

        # Re-assert the interactive backend here: modules imported after our
        # initial matplotlib.use() call (e.g. the database or train_code
        # modules) may have silently reset it to a non-interactive backend
        # like "Agg" for their own headless plotting needs.
        current_backend = matplotlib.get_backend()
        print(f"[backend] Backend right before opening figure: {current_backend}")
        if current_backend.lower() == "agg":
            print("[backend] Backend was reset to Agg by a later import — forcing it back to an interactive one.")
            for _bk in _BACKENDS_TO_TRY:
                try:
                    plt.switch_backend(_bk)
                    print(f"[backend] Switched to: {matplotlib.get_backend()}")
                    break
                except Exception as exc:
                    print(f"[backend]   {_bk} failed: {exc}")

        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        # Bring the window to the front on Windows/Qt where it can otherwise
        # open behind other windows.
        try:
            manager = self.fig.canvas.manager
            if hasattr(manager, "window"):
                manager.window.raise_()
                manager.window.activateWindow()
        except Exception:
            pass
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    def _on_key(self, event):
        if event.key == "right":
            self.idx = min(self.idx + 1, self.n_total - 1)
            self._draw()
        elif event.key == "left":
            self.idx = max(self.idx - 1, 0)
            self._draw()
        elif event.key == "q":
            plt.close(self.fig)

    def _draw(self):
        self.ax.clear()
        sample_row = self.df.iloc[self.idx]
        label = sample_row["final_label"]
        colour = LABEL_COLOURS.get(label, DEFAULT_COLOUR)
        file_path = self.raw_dir / str(sample_row["raw_data_file"])

        header = (
            f"Solutions: {self.solutions_str}  |  "
            f"Sample {self.idx + 1}/{self.n_total}  |  "
            f"Label: {label}  |  ID: {sample_row.get('id', '?')}  |  "
            f"V: {sample_row.get('target_voltage', '?')}  |  "
            f"Flow: {sample_row.get('flow_rate', '?')}"
        )

        if not file_path.exists():
            self.ax.text(0.5, 0.5, "file not found", ha="center", va="center", color="red", fontsize=12)
            self.ax.set_title(header, fontsize=10)
        else:
            try:
                raw, filtered = load_and_filter_waveform(file_path, self.cutoff_hz, float(sample_row["sample_rate"]))
                t_ms = build_time_axis(len(raw), float(sample_row["sample_rate"]))
                self.ax.plot(t_ms, raw, color=colour, alpha=0.35, linewidth=0.8, label="Raw")
                if SHOW_FILTERED:
                    self.ax.plot(t_ms, filtered, color=colour, alpha=0.9, linewidth=1.5, label="Filtered")

                self.ax.set_title(header, fontsize=10)
                self.ax.set_xlabel("Time (ms)", fontsize=9)
                self.ax.set_ylabel("Current (nA)", fontsize=9)
                self.ax.grid(True, linestyle="--", alpha=0.5)
                self.ax.legend(fontsize=8, loc="upper right")

                if Y_LIMITS is not None:
                    self.ax.set_ylim(*Y_LIMITS)
            except Exception as exc:
                self.ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center", color="red", fontsize=9, wrap=True)
                self.ax.set_title(header, fontsize=10)

        self.fig.suptitle("Use ← / → to navigate, q to quit", fontsize=9, y=0.99)
        self.fig.canvas.draw_idle()

    def show(self):
        # block=True ensures the script waits here until the window is closed,
        # instead of returning immediately (which is what makes it look like
        # "nothing opened").
        plt.show(block=True)


def main():
    BASE = Path("data")
    RAW_DIR = BASE / "raw_waveforms"

    print(f"[1/4] Loading database from: {BASE}")
    db = ElectrosprayDatabase(str(BASE))

    # db.load_training_dataframe() will interactively ask for solution(s)
    df_db = db.load_training_dataframe()

    if df_db.empty:
        raise ValueError("No data returned from database.")

    SOLUTION_COL = "solution_name"
    solutions_selected = []
    if SOLUTION_COL in df_db.columns:
        solutions_selected = df_db[SOLUTION_COL].dropna().unique().tolist()
        solutions_str = ", ".join(solutions_selected)
    else:
        solutions_str = "Unknown"

    print(f"[2/4] Found {len(df_db)} samples across selected solutions: {solutions_str}")
    df_db["final_label"] = df_db.apply(resolve_label, axis=1)

    # ── Filter out invalid labels ─────────────────────────────────────────────
    df_valid = df_db[
        df_db["final_label"].notna() &
        (~df_db["final_label"].isin(INVALID_LABELS))
    ].copy()

    if df_valid.empty:
        raise ValueError("All samples for the selected solutions have invalid/missing labels.")

    print(f"       {len(df_valid)} samples have valid labels.")

    # ── Sort in order (by id if available, else by index) — no randomization ──
    if "id" in df_valid.columns:
        df_valid = df_valid.sort_values(by="id")
    else:
        df_valid = df_valid.sort_index()

    if N_SAMPLES is not None:
        df_valid = df_valid.iloc[:N_SAMPLES]

    print(f"[3/4] Browsing {len(df_valid)} samples in order.")


    print("[4/4] Opening interactive viewer — use ← / → to navigate, q to quit.")
    browser = WaveformBrowser(df_valid, RAW_DIR, CUTOFF_HZ, solutions_str)
    browser.show()

    print("\nDone.")


if __name__ == "__main__":
    main()