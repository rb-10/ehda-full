"""
EHDA Feature Extraction Pipeline
==================================
Extracts time-domain, frequency-domain, and time-frequency features
from current signals for spray mode classification.

Sampling rate: 100,000 Hz (50,000 samples over 0.5 seconds)

Signal normalization (Layer 1) is applied to the raw current signal
before feature extraction. This removes solution-dependent amplitude
so that features reflect waveform SHAPE, not absolute magnitude.
Import normalize_signal from ehda_normalization to change the method.
"""

import json
import numpy as np
import pandas as pd
import pywt
from scipy import signal, stats
from pathlib import Path
from typing import Union
import warnings
warnings.filterwarnings("ignore")

# Layer 1 signal normalization — import here to keep extraction self-contained
def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalize: mean=0, std=1. Applied before all feature extraction."""
    mu, sigma = np.mean(x), np.std(x)
    return (x - mu) / sigma if sigma > 0 else x - mu

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
FS = 100_000          # Sampling frequency in Hz
WAVELET = "db4"       # Daubechies 4 — good for transient/pulse signals
WAVELET_LEVEL = 6     # Decomposition levels (~1.5 Hz resolution at level 6)

# Frequency bands (Hz) — tune these to your known EHDA physics
FREQ_BANDS = {
    "very_low":  (0,    50),     # Drop formation / dripping regime
    "low":       (50,   500),    # Slow oscillations / pulsation
    "mid":       (500,  5_000),  # Cone-jet oscillations
    "high":      (5_000, 20_000),# Fast instabilities / multi-jet
    "very_high": (20_000, 50_000) # Noise floor / high-freq artifacts
}


# ─────────────────────────────────────────────
# TIME-DOMAIN FEATURES
# ─────────────────────────────────────────────
def extract_time_domain(x: np.ndarray, precomputed: dict = None) -> dict:
    """
    Extract time-domain statistical features from the current signal.
    Uses precomputed values from JSON where available to avoid redundancy.
    """
    feats = {}
    n = len(x)

    # --- Already in your JSON (use precomputed if available) ---
    feats["mean"]      = precomputed.get("mean",      float(np.mean(x)))
    feats["std"]       = precomputed.get("deviation",  float(np.std(x)))
    feats["median"]    = precomputed.get("median",     float(np.median(x)))
    feats["rms"]       = precomputed.get("rms",        float(np.sqrt(np.mean(x**2))))
    feats["variance"]  = precomputed.get("variance",   float(np.var(x)))

    # --- New time-domain features ---
    peak = float(np.max(np.abs(x)))
    feats["peak"]            = peak
    feats["crest_factor"]    = peak / feats["rms"] if feats["rms"] > 0 else 0.0
    feats["kurtosis"]        = float(stats.kurtosis(x, fisher=True))   # excess kurtosis (0 = Gaussian)
    feats["skewness"]        = float(stats.skew(x))                    # asymmetry of distribution
    feats["peak_to_peak"]    = float(np.max(x) - np.min(x))
    feats["iqr"]             = float(np.percentile(x, 75) - np.percentile(x, 25))  # robust spread

    # Derivative variance — how rapidly/erratically the signal changes
    dx = np.diff(x)
    feats["derivative_variance"] = float(np.var(dx))
    feats["derivative_mean_abs"] = float(np.mean(np.abs(dx)))  # mean rate of change

    # Zero-crossing rate — how often the (demeaned) signal crosses zero
    x_centered = x - np.mean(x)
    feats["zero_crossing_rate"] = float(np.sum(np.diff(np.sign(x_centered)) != 0) / n)

    # Shape factor — RMS / mean absolute value (sensitive to waveform shape)
    mean_abs = np.mean(np.abs(x))
    feats["shape_factor"] = feats["rms"] / mean_abs if mean_abs > 0 else 0.0

    return feats


# ─────────────────────────────────────────────
# FREQUENCY-DOMAIN FEATURES
# ─────────────────────────────────────────────
def extract_frequency_domain(x: np.ndarray, fs: int = FS) -> dict:
    """
    Extract frequency-domain features using Welch's PSD estimate.
    Welch's method is more stable than a plain FFT for noisy signals.
    """
    feats = {}

    # Welch PSD — uses overlapping windows to reduce noise variance
    freqs, psd = signal.welch(x, fs=fs, nperseg=4096, noverlap=2048, window="hann")
    total_power = np.sum(psd)

    # Dominant frequency — frequency with the highest power
    feats["dominant_freq"] = float(freqs[np.argmax(psd)])

    # Mean and median frequency (center of mass of spectrum)
    feats["mean_freq"]   = float(np.sum(freqs * psd) / total_power) if total_power > 0 else 0.0
    feats["median_freq"] = float(freqs[np.searchsorted(np.cumsum(psd), total_power / 2)])

    # Band power — total power in each frequency band
    for band_name, (f_lo, f_hi) in FREQ_BANDS.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        feats[f"band_power_{band_name}"] = float(np.sum(psd[mask]))
        # Relative band power (fraction of total) — normalises for signal amplitude
        feats[f"band_power_{band_name}_rel"] = (
            float(np.sum(psd[mask]) / total_power) if total_power > 0 else 0.0
        )

    # Spectral entropy — how spread the energy is across frequencies
    # Low = concentrated (ordered), High = spread (chaotic)
    psd_norm = psd / total_power if total_power > 0 else psd + 1e-12
    psd_norm = np.clip(psd_norm, 1e-12, None)
    feats["spectral_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm)))

    # Spectral centroid spread (bandwidth) — width of the dominant spectral region
    feats["spectral_bandwidth"] = float(
        np.sqrt(np.sum(((freqs - feats["mean_freq"]) ** 2) * psd) / total_power)
        if total_power > 0 else 0.0
    )

    # Spectral rolloff — frequency below which 85% of energy lies
    cumulative = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumulative, 0.85 * total_power)
    feats["spectral_rolloff"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Total signal power
    feats["total_power"] = float(total_power)

    return feats


# ─────────────────────────────────────────────
# TIME-FREQUENCY FEATURES (WAVELETS)
# ─────────────────────────────────────────────
def extract_wavelet_features(x: np.ndarray,
                              wavelet: str = WAVELET,
                              level: int = WAVELET_LEVEL) -> dict:
    """
    Discrete Wavelet Transform (DWT) decomposition.

    Each level captures a frequency band:
      Level 1 detail:  25,000 – 50,000 Hz
      Level 2 detail:  12,500 – 25,000 Hz
      Level 3 detail:   6,250 – 12,500 Hz
      Level 4 detail:   3,125 –  6,250 Hz
      Level 5 detail:   1,563 –  3,125 Hz
      Level 6 detail:     781 –  1,563 Hz
      Level 6 approx:       0 –    781 Hz
    """
    feats = {}
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)

    total_energy = sum(np.sum(c**2) for c in coeffs)

    for i, c in enumerate(coeffs):
        label = f"wt_approx_L{level}" if i == 0 else f"wt_detail_L{level + 1 - i}"
        energy = float(np.sum(c**2))
        feats[f"{label}_energy"]     = energy
        feats[f"{label}_energy_rel"] = energy / total_energy if total_energy > 0 else 0.0
        feats[f"{label}_std"]        = float(np.std(c))
        feats[f"{label}_kurtosis"]   = float(stats.kurtosis(c, fisher=True))

    feats["wavelet_total_energy"] = float(total_energy)

    return feats


# ─────────────────────────────────────────────
# METADATA FEATURES
# ─────────────────────────────────────────────
def extract_metadata(sample: dict) -> dict:
    """
    Extract non-signal features from the sample record.
    These are physical operating conditions — very informative for mode boundaries.
    """
    return {
        "target_voltage": float(sample.get("target_voltage", 0.0)),
        "actual_voltage":  float(sample.get("voltage", 0.0)),
        "voltage_error":   float(sample.get("voltage", 0.0)) - float(sample.get("target_voltage", 0.0)),
        "flow_rate":       float(sample.get("flow_rate", 0.0)),
        "current_PS":      float(sample.get("current_PS", 0.0)),
    }


# ─────────────────────────────────────────────
# MAIN EXTRACTION FUNCTION
# ─────────────────────────────────────────────
def extract_features(sample: dict, normalize_signal: bool = False) -> dict:
    """
    Extract all features from a single sample dict.
    Returns a flat feature dictionary ready for a DataFrame row.

    Parameters
    ----------
    normalize_signal : bool (default False)
        If True, z-score the raw current signal before feature extraction.
        This makes shape-based features solution-agnostic (useful when
        training across multiple solutions with very different conductivities),
        but destroys all amplitude information — mean, std, rms, variance,
        and absolute band powers become ~constant across every sample.
        Leave False (default) unless you have a specific cross-solution
        generalization need; amplitude features are strong spray-mode
        discriminators within a single solution.
    """
    x_raw = np.array(sample["current"], dtype=np.float64)

    # ── Layer 1: Signal normalization (optional) ────────────────────────────
    if normalize_signal:
        x = _zscore(x_raw)
        # After z-score: mean≈0, std≈1 — precomputed JSON values no longer apply.
        precomputed = {}
    else:
        x = x_raw
        # Use precomputed values from JSON where available to avoid redundancy.
        precomputed = {
            "mean":     sample.get("mean"),
            "deviation": sample.get("deviation"),
            "median":   sample.get("median"),
            "rms":      sample.get("rms"),
            "variance": sample.get("variance"),
        }
        # Filter out None values so extract_time_domain falls back to computing them
        precomputed = {k: v for k, v in precomputed.items() if v is not None}

    features = {}
    features["sample_id"] = sample.get("id")
    # All labels stored under image_classification; fall back to spray_mode if absent
    features["label"]     = sample.get("image_classification", sample.get("spray_mode"))
    features["timestamp"] = sample.get("timestamp")

    features.update(extract_metadata(sample))
    features.update(extract_time_domain(x, precomputed))
    features.update(extract_frequency_domain(x))
    features.update(extract_wavelet_features(x))

    return features


# ─────────────────────────────────────────────
# PROCESS JSON FILE(S)
# ─────────────────────────────────────────────

def _parse_samples(data, filepath):
    """
    Detect JSON structure and return a flat list of sample dicts.

    Supported formats:
      A) List of samples:
            [ {"id": 1, ...}, {"id": 2, ...} ]

      B) Dict of named samples (your format):
            { "sample 1": {"id": 1, ...}, "sample 2": {"id": 2, ...} }

      C) Dict with a "samples" or "data" wrapper key:
            { "samples": [ {"id": 1, ...}, ... ] }

      D) Single sample dict:
            { "id": 1, "current": [...], ... }
    """
    if isinstance(data, list):
        return data  # Format A

    if isinstance(data, dict):
        if "samples" in data:
            return data["samples"]  # Format C
        if "data" in data:
            return data["data"]     # Format C (alt key)

        # Format B — dict of named samples, possibly with non-sample keys like "_meta".
        # Collect every value that looks like a sample (has a "current" key).
        # Non-sample keys (metadata, config, etc.) are silently skipped.
        samples = []
        for key, val in data.items():
            if isinstance(val, dict) and "current" in val:
                val.setdefault("sample_key", key)
                samples.append(val)
        if samples:
            return samples

        if "current" in data:
            return [data]  # Format D — single sample

    raise ValueError(
        f"Cannot parse JSON structure in {Path(filepath).name}.\n"
        f"Expected a list, dict-of-samples, or dict with a 'samples' key.\n"
        f"Top-level keys found: {list(data.keys())[:8]}"
    )


def process_json_file(filepath, normalize_signal: bool = False):
    """Load a single JSON file and extract features from all samples."""
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        data = json.load(f)

    samples = _parse_samples(data, filepath)
    total = len(samples)

    rows = []
    for i, sample in enumerate(samples):
        key  = sample.get("sample_key", f"#{i+1}")
        sid  = sample.get("id", "?")
        mode = sample.get("image_classification", sample.get("spray_mode", "?"))
        print(f"  [{filepath.name}]  {key}  (id={sid}, mode={mode})  [{i+1}/{total}]")
        try:
            row = extract_features(sample, normalize_signal=normalize_signal)
            row["source_file"] = filepath.name
            rows.append(row)
        except Exception as e:
            print(f"    \u26a0 Skipped {key} (id={sid}): {e}")

    return pd.DataFrame(rows)


def process_multiple_files(file_pattern: str = "*.json",
                           folder: Union[str, Path] = ".",
                           normalize_signal: bool = False) -> pd.DataFrame:
    """Process all JSON files matching a pattern in a folder.

    Parameters
    ----------
    normalize_signal : bool (default False)
        Passed to extract_features(). See its docstring for the trade-off.
        Leave False unless you are training across multiple solutions.
    """
    folder = Path(folder)
    # Use rglob to recursively find all JSON files in subfolders, ignoring .venv
    files = [f for f in sorted(folder.rglob(file_pattern)) if ".venv" not in f.parts]
    print(f"Found {len(files)} JSON file(s) in {folder} and subfolders")

    dfs = []
    for fp in files:
        print(f"\nProcessing: {fp.relative_to(folder)}")
        dfs.append(process_json_file(fp, normalize_signal=normalize_signal))

    if not dfs:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {folder} or its subfolders")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nSUCCESS Total samples extracted: {len(combined)}")
    print(f"SUCCESS Total features per sample: {combined.shape[1]}")
    print(f"SUCCESS Label distribution:\n{combined['label'].value_counts()}")
    return combined


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # --- Single file ---
    # df = process_json_file("your_data.json")

    # --- Folder of files ---
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    df = process_multiple_files("*.json", folder=folder)

    # Save feature matrix
    out_path = Path(folder) / "ehda_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSUCCESS Saved to: {out_path}")

    # Quick summary
    feature_cols = [c for c in df.columns
                    if c not in ("sample_id", "label", "timestamp", "source_file")]
    print(f"\nFeature groups:")
    print(f"  Metadata:          {sum(1 for c in feature_cols if c in ['target_voltage','actual_voltage','voltage_error','flow_rate','current_PS'])}")
    print(f"  Time-domain:       {sum(1 for c in feature_cols if not c.startswith(('band_', 'spectral', 'dominant', 'mean_freq', 'median_freq', 'total_power', 'wt_')))}")
    print(f"  Frequency-domain:  {sum(1 for c in feature_cols if c.startswith(('band_', 'spectral', 'dominant', 'mean_freq', 'median_freq', 'total_power')))}")
    print(f"  Wavelet:           {sum(1 for c in feature_cols if c.startswith('wt_'))}")
    print(f"  TOTAL:             {len(feature_cols)}")