"""
EHDA Normalization Pipeline (Enhanced with Custom Multi-Strategy Scaler)
========================================================================
Two-layer normalization strategy to make features solution-agnostic:

  Layer 1 — Signal normalization (applied to raw current before feature extraction)
             Removes absolute amplitude so shape-based features are comparable
             across solutions with different conductivities.

  Layer 2 — Feature normalization (applied to extracted feature matrix)
             Custom multi-strategy approach with 5 different normalization types:
               • LINEAR:       divide by domain-specific factors
               • LOG+ROBUST:   signed log transform + RobustScaler
               • ROBUST:       RobustScaler (median/IQR based) for outliers
               • PASSTHROUGH:  leave as-is (already normalized)
               • LOW-VARIANCE: don't scale (already mean≈0, std≈1)

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
NON_FEATURE_COLS = ["sample_id", "label", "timestamp", "source_file", "is_clean_label"]


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
    Fits and applies custom multi-strategy normalization.

    Uses 5 different strategies based on feature characteristics:
      1. LINEAR: divide by domain-specific factors (e.g., voltage / 10000)
      2. LOG + ROBUST: signed log transform + RobustScaler (heavy-tailed)
      3. ROBUST: RobustScaler only (outlier-prone)
      4. PASSTHROUGH: leave unchanged (already normalized)
      5. LOW-VARIANCE: don't scale (already mean≈0, std≈1)

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
        self.amplitude_cols: list = []
        self.metadata_cols:  list = []
        self.invariant_cols: list = []
        self.fitted = False
        
        # Internal multi-strategy scalers
        self._linear_factors = {}
        self._log_robust_scaler = RobustScaler()
        self._robust_scaler = RobustScaler()
        self._classification = {}

    def fit(self, df: pd.DataFrame) -> "EHDAFeatureNormalizer":
        """Fit scalers on training data. Call once, on training set only."""
        # Keep backward compatibility with original interface
        self.amplitude_cols, self.metadata_cols, self.invariant_cols, _ = \
            _classify_columns(df)

        # Also classify using custom strategies
        self._classification = self._classify_columns_custom(df)
        
        # Fit linear factors (just store them)
        linear_cols = self._classification["linear"]
        for col in linear_cols:
            if col in self._LINEAR_FEATURES:
                self._linear_factors[col] = self._LINEAR_FEATURES[col]
            else:
                self._linear_factors[col] = 1.0

        # Fit log + robust scaler
        log_robust_cols = self._classification["log_robust"]
        if log_robust_cols:
            X_log = self._signed_log1p(df[log_robust_cols].values)
            self._log_robust_scaler.fit(X_log)

        # Fit robust scaler
        robust_cols = self._classification["robust"]
        if robust_cols:
            self._robust_scaler.fit(df[robust_cols].values)

        self.fitted = True
        print(f"SUCCESS Normalizer fitted on {len(df)} samples")
        print(f"  Strategy 1 - LINEAR (÷ factor):        {len(linear_cols)}")
        print(f"  Strategy 2 - LOG + ROBUST:             {len(log_robust_cols)}")
        print(f"  Strategy 3 - ROBUST (RobustScaler):    {len(robust_cols)}")
        print(f"  Strategy 4 - PASSTHROUGH (unchanged):  {len(self._classification['passthrough'])}")
        print(f"  Strategy 5 - LOW-VARIANCE (unchanged): {len(self._classification['low_variance'])}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted normalization to any DataFrame (train or new data)."""
        if not self.fitted:
            raise RuntimeError("Call .fit() before .transform()")

        df_out = df.copy()

        # Strategy 1: LINEAR
        linear_cols = self._classification["linear"]
        for col in linear_cols:
            factor = self._linear_factors.get(col, 1.0)
            df_out[col] = df[col] / factor

        # Strategy 2: LOG + ROBUST
        log_robust_cols = self._classification["log_robust"]
        if log_robust_cols:
            X_log = self._signed_log1p(df[log_robust_cols].values)
            X_scaled = self._log_robust_scaler.transform(X_log)
            df_out[log_robust_cols] = X_scaled

        # Strategy 3: ROBUST
        robust_cols = self._classification["robust"]
        if robust_cols:
            X_scaled = self._robust_scaler.transform(df[robust_cols].values)
            df_out[robust_cols] = X_scaled

        # Strategy 4 & 5: PASSTHROUGH and LOW-VARIANCE (already handled by copy)
        
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        return self.fit(df).transform(df)

    def get_feature_columns(self) -> list:
        """Return the ordered list of columns to feed to the model."""
        return self.amplitude_cols + self.metadata_cols + self.invariant_cols

    def save(self, folder: str = "current_classification/scalers") -> None:
        """Persist fitted scalers to disk for inference reuse."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._linear_factors, folder / "linear_factors.pkl")
        joblib.dump(self._log_robust_scaler, folder / "log_robust_scaler.pkl")
        joblib.dump(self._robust_scaler, folder / "robust_scaler.pkl")
        joblib.dump({
            "amplitude_cols": self.amplitude_cols,
            "metadata_cols":  self.metadata_cols,
            "invariant_cols": self.invariant_cols,
            "fitted":         self.fitted,
            "classification": self._classification,
        }, folder / "normalizer_meta.pkl")
        print(f"SUCCESS Scalers saved to {folder}/")

    @classmethod
    def load(cls, folder: str = "current_classification/scalers") -> "EHDAFeatureNormalizer":
        """Load previously fitted scalers from disk."""
        folder = Path(folder)
        obj = cls()
        obj._linear_factors = joblib.load(folder / "linear_factors.pkl")
        obj._log_robust_scaler = joblib.load(folder / "log_robust_scaler.pkl")
        obj._robust_scaler = joblib.load(folder / "robust_scaler.pkl")
        meta = joblib.load(folder / "normalizer_meta.pkl")
        obj.amplitude_cols = meta["amplitude_cols"]
        obj.metadata_cols  = meta["metadata_cols"]
        obj.invariant_cols = meta["invariant_cols"]
        obj.fitted         = meta["fitted"]
        obj._classification = meta["classification"]
        print(f"SUCCESS Scalers loaded from {folder}/")
        return obj

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPER METHODS (custom multi-strategy logic)
    # ─────────────────────────────────────────────────────────────────────────

    # Strategy 1: Linear features with domain-specific factors
    _LINEAR_FEATURES = {
        "target_voltage": 10000.0,
        "actual_voltage": 10000.0,
        "flow_rate": 50.0,
        "dominant_freq": 100.0,
    }

    # Strategy 2: Heavy-tailed features (log + robust)
    _LOG_ROBUST_FEATURES = [
        "total_power", "wavelet_total_energy",
        "wt_approx_L6_energy",
        "wt_detail_L6_energy", "wt_detail_L5_energy",
        "wt_detail_L4_energy", "wt_detail_L3_energy",
        "wt_detail_L2_energy", "wt_detail_L1_energy",
        "mean_freq", "median_freq",
        "spectral_rolloff", "spectral_bandwidth",
        "derivative_variance",
    ]

    # Strategy 3: Robust features (outlier-prone)
    _ROBUST_FEATURES = [
        "voltage_error",
        "median", "iqr",
        "derivative_mean_abs",
        "zero_crossing_rate",
        "shape_factor",
        "skewness", "kurtosis",
        "peak", "peak_to_peak", "crest_factor",
    ]

    # Strategy 4: Passthrough (already normalized)
    _PASSTHROUGH_PATTERN = "_rel"

    # Strategy 5: Low-variance (don't scale)
    _LOW_VARIANCE_FEATURES = [
        "std", "rms", "variance",
        "mean", "current_PS",
    ]

    @staticmethod
    def _signed_log1p(x: np.ndarray) -> np.ndarray:
        """Apply signed logarithm: sign(x) * log(1 + |x|)"""
        return np.sign(x) * np.log1p(np.abs(x))

    def _classify_columns_custom(self, df: pd.DataFrame) -> dict:
        """Classify columns into the 5 custom scaling strategies."""
        all_cols = list(df.columns)
        classification = {
            "linear": [],
            "log_robust": [],
            "robust": [],
            "passthrough": [],
            "low_variance": [],
            "skip": [],
        }

        for col in all_cols:
            if col in NON_FEATURE_COLS:
                classification["skip"].append(col)
            elif col in self._LINEAR_FEATURES:
                classification["linear"].append(col)
            elif col in self._LOG_ROBUST_FEATURES:
                classification["log_robust"].append(col)
            elif col in self._ROBUST_FEATURES:
                classification["robust"].append(col)
            elif col.endswith(self._PASSTHROUGH_PATTERN):
                classification["passthrough"].append(col)
            elif col in self._LOW_VARIANCE_FEATURES:
                classification["low_variance"].append(col)
            else:
                # Default: treat as robust
                classification["robust"].append(col)

        return classification


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE HELPER
# Combines both layers into one clean call
# ─────────────────────────────────────────────────────────────────────────────

def prepare_training_data(
    df: pd.DataFrame,
    signal_norm_method: str = "zscore",
    scaler_save_path: str = "current_classification/scalers",
    drop_metadata: bool = False,
    exclude_label: str = "EXCLUDE",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list, "EHDAFeatureNormalizer"]:
    """
    Prepare raw (un-normalized) feature arrays for training.

    The normalizer is intentionally NOT fitted here to prevent data leakage.
    Fitting happens inside train() after the train/test split, so the scaler
    only ever sees training-set statistics. The returned normalizer carries
    column-type metadata (which columns are amplitude, metadata, invariant)
    and is passed to train() to be fitted there.

    Parameters
    ----------
    df              : DataFrame from feature_extraction.py
    signal_norm_method : for documentation only — apply in extraction step
    scaler_save_path   : passed to train() so it knows where to save scalers
    drop_metadata   : if True, exclude operating-condition columns (voltage,
                      flow_rate, etc.) — recommended for cross-solution models
    exclude_label   : rows with this label are dropped before training

    Returns
    -------
    df              : filtered raw DataFrame (not normalized)
    X               : raw feature matrix as numpy array
    labels          : string label array
    feature_names   : ordered list of feature column names
    normalizer      : EHDAFeatureNormalizer with column metadata set but NOT
                      fitted — pass this to train() to fit on X_train only
    """
    # Drop any samples with missing labels or excluded label
    df = df.dropna(subset=["label"]).copy()
    df = df[(df["label"] != "N/A") & (df["label"] != exclude_label)].copy()

    # Classify columns into groups — but do NOT fit the scalers yet.
    # Fitting on the full dataset would leak test-set statistics into the scaler.
    amplitude_cols, metadata_cols, invariant_cols, _ = _classify_columns(df)

    # Build the ordered feature list
    all_feature_cols = amplitude_cols + metadata_cols + invariant_cols
    if drop_metadata:
        all_feature_cols = [f for f in all_feature_cols if f not in METADATA_FEATURES]

    # Pre-populate normalizer column lists so train() knows how to scale each column.
    # Only include columns that are actually in the selected feature set.
    normalizer = EHDAFeatureNormalizer()
    normalizer.amplitude_cols = [c for c in amplitude_cols if c in all_feature_cols]
    normalizer.metadata_cols  = [c for c in metadata_cols  if c in all_feature_cols]
    normalizer.invariant_cols = [c for c in invariant_cols if c in all_feature_cols]
    normalizer._scaler_save_path = scaler_save_path  # carry through for train()

    feature_names = all_feature_cols
    X      = df[feature_names].values
    labels = df["label"].values

    print(f"\nSUCCESS Raw training data ready: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  NOTE: normalizer is unfitted — it will be fitted on X_train inside train()")
    print(f"  Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u:<20} {c} samples")

    return df, X, labels, feature_names, normalizer


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
    from feature_extraction import process_json_file, process_multiple_files

    df = process_multiple_files("*.json", folder=r"C:\Users\HV\Desktop\bruno_work\main\data\current_training")
    folder=r"C:\Users\HV\Desktop\bruno_work\main\data\current_training"
    df_norm, X, labels, feature_names, normalizer = prepare_training_data(
        df, scaler_save_path="scalers"
    )

    print(f"\nSample of normalized values (first sample):")
    for name, val in zip(feature_names[:10], X[0, :10]):
        print(f"  {name:<40} {val:.4f}")
    print("  ...")

    out_path = Path(folder) / "ehda_features_normalized.csv"
    df_norm.to_csv(out_path, index=False)
    print(f"\nSUCCESS Normalized features saved to: {out_path}")