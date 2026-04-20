"""
EHDA Normalization Pipeline
============================
Two-layer normalization strategy to make features solution-agnostic:

  Layer 1 — Signal normalization (applied to raw current before feature extraction)
             Removes absolute amplitude so shape-based features are comparable
             across solutions with different conductivities.

  Layer 2 — Feature normalization (applied to extracted feature matrix)
             Different strategies per feature type:
               • Amplitude-carrying features   → RobustScaler (median/IQR based)
               • Already-invariant features    → left as-is (documented below)
               • Metadata / operating params   → StandardScaler (they should vary)

This module also handles saving and loading the fitted scalers so that
at inference time you apply the exact same transformation as at training time.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE TAXONOMY
# Which features carry absolute amplitude and need scaling,
# and which are already dimensionless / solution-invariant.
# ─────────────────────────────────────────────────────────────────────────────

# These carry absolute current amplitude → scale with RobustScaler
# RobustScaler uses median and IQR, so it's resistant to outlier samples.
AMPLITUDE_FEATURES = [
    "mean", "std", "median", "rms", "variance",
    "peak", "peak_to_peak", "iqr",
    "derivative_variance", "derivative_mean_abs",
    "total_power",
    "wavelet_total_energy",
    # Absolute (non-relative) band powers carry amplitude info
    "band_power_very_low", "band_power_low", "band_power_mid",
    "band_power_high", "band_power_very_high",
    # Absolute wavelet energies and stds per level
    # (generated dynamically — matched by prefix below)
]
AMPLITUDE_PREFIXES = ("wt_approx_", "wt_detail_")  # _energy and _std columns
AMPLITUDE_SUFFIXES = ("_energy", "_std")            # but NOT _energy_rel or _kurtosis

# These are already dimensionless / amplitude-invariant — do NOT scale
# (scaling them could actually destroy their meaning)
INVARIANT_FEATURES = [
    "crest_factor",          # ratio: peak / RMS
    "shape_factor",          # ratio: RMS / mean_abs
    "kurtosis",              # 4th statistical moment (dimensionless)
    "skewness",              # 3rd statistical moment (dimensionless)
    "zero_crossing_rate",    # count / N (dimensionless)
    "spectral_entropy",      # information-theoretic (dimensionless)
    "dominant_freq",         # Hz — invariant to amplitude, variant to mode
    "mean_freq",             # Hz
    "median_freq",           # Hz
    "spectral_bandwidth",    # Hz
    "spectral_rolloff",      # Hz
    # Relative band powers are fractions (sum to 1) — already amplitude-free
    "band_power_very_low_rel", "band_power_low_rel", "band_power_mid_rel",
    "band_power_high_rel", "band_power_very_high_rel",
    # Relative wavelet energies (fractions) and kurtosis per level
    # (matched by suffix below)
]
INVARIANT_SUFFIXES = ("_energy_rel", "_kurtosis")

# Operating condition metadata — standardise separately
METADATA_FEATURES = [
    "target_voltage", "actual_voltage", "voltage_error",
    "flow_rate", "current_PS",
]

# Columns that are never fed to the model
NON_FEATURE_COLS = ["sample_id", "label", "timestamp", "source_file"]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — SIGNAL-LEVEL NORMALIZATION
# Applied to the raw current array BEFORE feature extraction.
# ─────────────────────────────────────────────────────────────────────────────

def normalize_signal(x: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize a raw current signal to remove solution-dependent amplitude.

    Parameters
    ----------
    x : raw current array (50,000 samples)
    method : one of
        "zscore"  — subtract mean, divide by std. Signal becomes mean=0, std=1.
                    Best for most cases. Preserves waveform shape perfectly.
        "robust"  — subtract median, divide by IQR. Resistant to large spikes.
                    Better if dripping pulses create extreme outliers.
        "minmax"  — scale to [0, 1]. Not recommended: sensitive to single spikes.

    Returns
    -------
    Normalized signal array of the same length.

    Why this helps
    --------------
    After z-score normalization, two signals that are identical in shape
    but differ only in amplitude (e.g., because one solution has higher
    conductivity) become numerically identical. Features that describe
    SHAPE (kurtosis, crest factor, spectral entropy, relative band powers,
    wavelet energy ratios) will be identical. Features that describe
    AMPLITUDE (mean, rms, variance, absolute band power) will now reflect
    within-solution dynamics rather than between-solution offsets.
    """
    if method == "zscore":
        mu, sigma = np.mean(x), np.std(x)
        return (x - mu) / sigma if sigma > 0 else x - mu

    elif method == "robust":
        med = np.median(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        return (x - med) / iqr if iqr > 0 else x - med

    elif method == "minmax":
        lo, hi = np.min(x), np.max(x)
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: zscore, robust, minmax")


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — FEATURE-LEVEL NORMALIZATION
# Applied to the extracted feature DataFrame.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_columns(df: pd.DataFrame) -> Tuple[list, list, list, list]:
    """
    Sort all feature columns into four buckets:
      amplitude_cols  → scale with RobustScaler
      metadata_cols   → scale with StandardScaler
      invariant_cols  → leave as-is
      skip_cols       → not model inputs
    """
    all_cols = list(df.columns)
    amplitude_cols, metadata_cols, invariant_cols, skip_cols = [], [], [], []

    for col in all_cols:
        if col in NON_FEATURE_COLS:
            skip_cols.append(col)
            continue
        if col in METADATA_FEATURES:
            metadata_cols.append(col)
            continue
        if col in INVARIANT_FEATURES:
            invariant_cols.append(col)
            continue
        if col.endswith(INVARIANT_SUFFIXES):
            invariant_cols.append(col)
            continue
        if col in AMPLITUDE_FEATURES:
            amplitude_cols.append(col)
            continue
        # Wavelet amplitude features matched by prefix + suffix
        if col.startswith(AMPLITUDE_PREFIXES) and col.endswith(AMPLITUDE_SUFFIXES):
            amplitude_cols.append(col)
            continue
        # Anything else — treat as amplitude to be safe
        amplitude_cols.append(col)

    return amplitude_cols, metadata_cols, invariant_cols, skip_cols


class EHDAFeatureNormalizer:
    """
    Fits and applies the two-scaler normalization strategy.

    Usage
    -----
    # Training time:
    normalizer = EHDAFeatureNormalizer()
    df_norm = normalizer.fit_transform(df_features)
    normalizer.save("scalers/")

    # Inference time:
    normalizer = EHDAFeatureNormalizer.load("scalers/")
    df_norm = normalizer.transform(df_new)
    """

    def __init__(self):
        self.amplitude_scaler = RobustScaler()    # median + IQR — robust to outliers
        self.metadata_scaler  = StandardScaler()  # mean + std — fine for op. conditions
        self.amplitude_cols: list = []
        self.metadata_cols:  list = []
        self.invariant_cols: list = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "EHDAFeatureNormalizer":
        """Fit scalers on training data. Call once, on training set only."""
        self.amplitude_cols, self.metadata_cols, self.invariant_cols, _ = \
            _classify_columns(df)

        if self.amplitude_cols:
            self.amplitude_scaler.fit(df[self.amplitude_cols])
        if self.metadata_cols:
            self.metadata_scaler.fit(df[self.metadata_cols])

        self.fitted = True
        print(f"✓ Normalizer fitted on {len(df)} samples")
        print(f"  Amplitude features (RobustScaler):   {len(self.amplitude_cols)}")
        print(f"  Metadata features  (StandardScaler): {len(self.metadata_cols)}")
        print(f"  Invariant features (untouched):      {len(self.invariant_cols)}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted normalization to any DataFrame (train or new data)."""
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform()")

        df_out = df.copy()

        if self.amplitude_cols:
            df_out[self.amplitude_cols] = self.amplitude_scaler.transform(
                df[self.amplitude_cols]
            )
        if self.metadata_cols:
            df_out[self.metadata_cols] = self.metadata_scaler.transform(
                df[self.metadata_cols]
            )
        # Invariant and skip columns are copied unchanged
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        return self.fit(df).transform(df)

    def get_feature_columns(self) -> list:
        """Return the ordered list of columns to feed to the model."""
        return self.amplitude_cols + self.metadata_cols + self.invariant_cols

    def save(self, folder: str = "scalers") -> None:
        """Persist fitted scalers to disk for inference reuse."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.amplitude_scaler, folder / "amplitude_scaler.pkl")
        joblib.dump(self.metadata_scaler,  folder / "metadata_scaler.pkl")
        joblib.dump({
            "amplitude_cols": self.amplitude_cols,
            "metadata_cols":  self.metadata_cols,
            "invariant_cols": self.invariant_cols,
            "fitted":         self.fitted,
        }, folder / "normalizer_meta.pkl")
        print(f"✓ Scalers saved to {folder}/")

    @classmethod
    def load(cls, folder: str = "scalers") -> "EHDAFeatureNormalizer":
        """Load previously fitted scalers from disk."""
        folder = Path(folder)
        obj = cls()
        obj.amplitude_scaler = joblib.load(folder / "amplitude_scaler.pkl")
        obj.metadata_scaler  = joblib.load(folder / "metadata_scaler.pkl")
        meta = joblib.load(folder / "normalizer_meta.pkl")
        obj.amplitude_cols = meta["amplitude_cols"]
        obj.metadata_cols  = meta["metadata_cols"]
        obj.invariant_cols = meta["invariant_cols"]
        obj.fitted         = meta["fitted"]
        print(f"✓ Scalers loaded from {folder}/")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE HELPER
# Combines both layers into one clean call
# ─────────────────────────────────────────────────────────────────────────────

def prepare_training_data(
    df: pd.DataFrame,
    signal_norm_method: str = "zscore",
    scaler_save_path: str = "scalers",
    drop_metadata: bool = False,
    exclude_label: str = "EXCLUDE",
) -> Tuple[pd.DataFrame, np.ndarray, list, "EHDAFeatureNormalizer"]:
    """
    Full pipeline from raw feature DataFrame to model-ready arrays.

    Note: signal normalization (Layer 1) should ideally be applied
    BEFORE feature extraction (in ehda_feature_extraction.py).
    This function handles Layer 2 (feature normalization).

    Parameters
    ----------
    df              : DataFrame from ehda_feature_extraction.py
    signal_norm_method : kept here for documentation; apply in extraction step
    scaler_save_path   : where to persist fitted scalers

    Returns
    -------
    df_norm         : normalized DataFrame (all columns)
    X               : feature matrix as numpy array (ready for model)
    feature_names   : list of feature column names (same order as X)
    normalizer      : fitted EHDAFeatureNormalizer (save this!)
    """

    # Drop any samples with missing labels or excluded label
    df = df.dropna(subset=["label"]).copy()
    df = df[(df["label"] != "N/A") & (df["label"] != exclude_label)].copy()

    normalizer = EHDAFeatureNormalizer()
    df_norm = normalizer.fit_transform(df)

    # Optionally drop metadata columns
    feature_names = normalizer.get_feature_columns()
    if drop_metadata:
        feature_names = [f for f in feature_names if f not in METADATA_FEATURES]
        keep_cols = feature_names + ["label"] + [c for c in NON_FEATURE_COLS if c in df_norm.columns and c != "label"]
        df_norm = df_norm[keep_cols]


    X = df_norm[feature_names].values
    labels = df_norm["label"].values
    # Save scalers so inference can use identical transforms
    normalizer.save(scaler_save_path)

    print(f"\n✓ Training data ready: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u:<20} {c} samples")

    return df_norm, X, labels, feature_names, normalizer


def prepare_inference_sample(
    features: dict,
    normalizer: "EHDAFeatureNormalizer",
) -> np.ndarray:
    """
    Transform a single sample's features for live inference.

    Parameters
    ----------
    features   : dict from extract_features() (single sample)
    normalizer : fitted EHDAFeatureNormalizer loaded from disk

    Returns
    -------
    x_norm : 1D numpy array ready for model.predict()
    """
    df = pd.DataFrame([features])
    df_norm = normalizer.transform(df)
    feature_names = normalizer.get_feature_columns()
    return df_norm[feature_names].values[0]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK DEMO
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from feature_extraction import process_json_file, process_multiple_files

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    df = process_multiple_files("*.json", folder=folder)

    df_norm, X, labels, feature_names, normalizer = prepare_training_data(
        df, scaler_save_path="scalers"
    )

    print(f"\nSample of normalized values (first sample):")
    for name, val in zip(feature_names[:10], X[0, :10]):
        print(f"  {name:<40} {val:.4f}")
    print("  ...")

    out_path = Path(folder) / "ehda_features_normalized.csv"
    df_norm.to_csv(out_path, index=False)
    print(f"\n✓ Normalized features saved to: {out_path}")