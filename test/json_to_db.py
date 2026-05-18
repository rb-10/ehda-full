"""
electrospray_to_db.py
─────────────────────────────────────────────────────────────────────────────
Converts a directory of Electrospray JSON experiment files into a SQLite
database.

For each sample the script will:
  1. Use statistics already present in the JSON when available.
  2. Re-compute missing stats (mean_na, band powers, qty_max, pct_max …)
     from the raw 'current' array using ElectrosprayDataProcessing.

Usage
─────
    python electrospray_to_db.py                        # uses defaults below
    python electrospray_to_db.py --input ./data --db experiments.db
    python electrospray_to_db.py --input ./data --db experiments.db --cutoff 20000

Filename conventions (optional, adjust parse_filename() to taste)
─────────────────────────────────────────────────────────────────
    <solution_name>_<hv_position>_<anything>.json
    e.g.  EtOH_pos_run01.json  →  solution="EtOH", hv_position="pos"
    Files that don't match fall back to solution_name=stem, hv_position=None.
"""

import argparse
import json
import logging
import re
import sqlite3
from pathlib import Path

import numpy as np
import pywt
from scipy import signal, stats as scipy_stats
from scipy.integrate import trapezoid
from scipy.signal import filtfilt

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Default configuration ────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = "test/json_files"
DEFAULT_DB_PATH   = "test/electrospray.db"
SAMPLE_RATE       = 1e5          # 100 kHz  (matches ElectrosprayDataProcessing)
FILTER_ORDER      = 4
FILTER_CUTOFF_HZ  = 20_000      # low-pass cutoff; override with --cutoff
SATURATION_THR    = 39_950      # ADC clips at ~80 V → 39 950 counts

# ─── SQL ──────────────────────────────────────────────────────────────────────
_CREATE = """
CREATE TABLE IF NOT EXISTS measurements (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp             TEXT,
    solution_name         TEXT,
    hv_position           TEXT,
    target_voltage        REAL,
    actual_voltage        REAL,
    actual_current_ps     REAL,
    flow_rate             REAL,
    mean_na               REAL,
    deviation_na          REAL,
    median_na             REAL,
    rms_na                REAL,
    variance_na           REAL,
    qty_max               INTEGER,
    pct_max               REAL,
    band_power_v_low      REAL,
    band_power_low        REAL,
    band_power_mid        REAL,
    band_power_high       REAL,
    band_power_v_high     REAL,
    rf_spray_mode         TEXT,
    xgb_spray_mode        TEXT,
    image_classification  TEXT,
    manual_classification TEXT,
    video_file            TEXT,
    raw_data_file         TEXT
);
"""

_INSERT = """
INSERT INTO measurements (
    timestamp, solution_name, hv_position, target_voltage, actual_voltage,
    actual_current_ps, flow_rate,
    mean_na, deviation_na, median_na, rms_na, variance_na,
    qty_max, pct_max,
    band_power_v_low, band_power_low, band_power_mid,
    band_power_high, band_power_v_high,
    rf_spray_mode, xgb_spray_mode, image_classification,
    manual_classification, video_file, raw_data_file
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


# ─── Signal-processing class (adapted from provided code) ─────────────────────
class ElectrosprayDataProcessing:
    """
    Processes a single current trace.

    Parameters
    ----------
    sample_rate : float
        ADC sample rate in Hz (default 100 000).
    """

    def __init__(self, sample_rate: float = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._clear()

    def _clear(self):
        self.mean_value       = 0.0
        self.variance         = 0.0
        self.stddev           = 0.0
        self.med              = 0.0
        self.rms              = 0.0
        self.datapoints_filtered = np.array([])
        self.psd_freqs           = np.array([])
        self.psd_welch           = np.array([])

    # ── Filtering ─────────────────────────────────────────────────────────────
    def calculate_filter(self, b_coef, a_coef, datapoints: np.ndarray):
        """Zero-phase low-pass filter to remove high-frequency noise."""
        self.datapoints_filtered = filtfilt(b_coef, a_coef, datapoints)

    # ── Time domain ───────────────────────────────────────────────────────────
    def calculate_statistics(self, data: np.ndarray):
        self.mean_value = float(np.mean(data))
        self.variance   = float(np.var(data))
        self.stddev     = float(np.std(data))
        self.med        = float(np.median(data))
        self.rms        = float(np.sqrt(np.mean(data ** 2)))

    def calculate_peaks_signal(self, data: np.ndarray, threshold: float = SATURATION_THR):
        """Returns (threshold, qty_saturated, pct_saturated)."""
        qty_max = int(np.sum(data >= threshold))
        pct_max = float((qty_max / len(data)) * 100)
        return threshold, qty_max, pct_max

    # ── Frequency domain ──────────────────────────────────────────────────────
    def calculate_power_spectral_density(self, data: np.ndarray):
        self.psd_freqs, self.psd_welch = signal.welch(
            data, fs=self.sample_rate, nperseg=4096, noverlap=2048, window="hann"
        )

    def calculate_band_powers(self) -> dict:
        bands = {
            "v_low":  (0,       50),
            "low":    (50,      500),
            "mid":    (500,     2_000),
            "high":   (2_000,   10_000),
            "v_high": (10_000,  self.sample_rate / 2),
        }
        result = {}
        for name, (lo, hi) in bands.items():
            idx = np.logical_and(self.psd_freqs >= lo, self.psd_freqs < hi)
            result[f"band_power_{name}"] = (
                float(trapezoid(self.psd_welch[idx], self.psd_freqs[idx]))
                if np.any(idx) else 0.0
            )
        return result

    # ── Convenience ───────────────────────────────────────────────────────────
    def get_db_features_dictionary(self) -> dict:
        bp = self.calculate_band_powers()
        return {
            "mean_na":      self.mean_value,
            "variance_na":  self.variance,
            "deviation_na": self.stddev,
            "median_na":    self.med,
            "rms_na":       self.rms,
            **bp,
        }


# ─── Helper utilities ─────────────────────────────────────────────────────────
def build_filter_coefficients(cutoff_hz: float, order: int = FILTER_ORDER):
    """Design a Butterworth low-pass filter and return (b, a) coefficients."""
    nyq = SAMPLE_RATE / 2
    normalized = min(cutoff_hz / nyq, 0.999)   # must be < 1
    b, a = signal.butter(order, normalized, btype="low", analog=False)
    return b, a


def parse_filename(path: Path) -> tuple[str, str | None]:
    """
    Extract (solution_name, hv_position) from the file stem.

    Expects:  <solution>_<hv_position>_<rest>.json
    Falls back to (stem, None) when the pattern doesn't match.
    """
    stem  = path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return stem, None


def _to_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def process_sample(
    sample: dict,
    b_coef: np.ndarray,
    a_coef: np.ndarray,
    proc: ElectrosprayDataProcessing,
) -> dict:
    """
    Given a raw sample dict from the JSON, return a flat feature dict
    ready for database insertion.

    Statistics that are already present in the JSON are used directly.
    Missing statistics are computed from the 'current' array.
    """
    proc._clear()

    raw_current = sample.get("current")
    has_raw     = isinstance(raw_current, (list, np.ndarray)) and len(raw_current) > 0

    # ── Run signal processing if raw data is available ────────────────────────
    if has_raw:
        datapoints = np.asarray(raw_current, dtype=np.float64)
        proc.calculate_filter(b_coef, a_coef, datapoints)
        proc.calculate_statistics(proc.datapoints_filtered)
        proc.calculate_power_spectral_density(proc.datapoints_filtered)
        _, qty_max, pct_max = proc.calculate_peaks_signal(datapoints)
        computed = proc.get_db_features_dictionary()
        computed["qty_max"] = qty_max
        computed["pct_max"] = pct_max
    else:
        computed   = {}
        qty_max    = None
        pct_max    = None
        log.debug("No raw current data – will fall back to JSON statistics where available.")

    # ── Helper: JSON value → computed fallback → None ─────────────────────────
    def pick(json_key, computed_key=None):
        """Return JSON value if present, else computed value."""
        v = sample.get(json_key)
        if v is not None:
            return v
        if computed_key:
            return computed.get(computed_key)
        return None

    # ── Map JSON keys → DB columns ────────────────────────────────────────────
    # Time-domain statistics: the JSON stores 'mean', 'deviation' etc. (no _na suffix)
    # while the DB uses 'mean_na', 'deviation_na' etc.
    features = {
        # Core measurement
        "timestamp":         sample.get("timestamp"),
        "target_voltage":    _to_float_or_none(sample.get("target_voltage")),
        "actual_voltage":    _to_float_or_none(sample.get("voltage")),
        "actual_current_ps": _to_float_or_none(sample.get("current_PS")),
        "flow_rate":         _to_float_or_none(sample.get("flow_rate")),

        # Time-domain stats: prefer JSON value, fall back to computed
        "mean_na":      pick("mean",      "mean_na"),
        "deviation_na": pick("deviation", "deviation_na"),
        "median_na":    pick("median",    "median_na"),
        "rms_na":       pick("rms",       "rms_na"),
        "variance_na":  pick("variance",  "variance_na"),

        # Saturation metrics (only computable from raw data)
        "qty_max": computed.get("qty_max", qty_max),
        "pct_max": computed.get("pct_max", pct_max),

        # Band powers (only computable from raw data)
        "band_power_v_low":  computed.get("band_power_v_low"),
        "band_power_low":    computed.get("band_power_low"),
        "band_power_mid":    computed.get("band_power_mid"),
        "band_power_high":   computed.get("band_power_high"),
        "band_power_v_high": computed.get("band_power_v_high"),

        # Classifications & metadata
        "rf_spray_mode":         sample.get("rf_spray_mode"),
        "xgb_spray_mode":        sample.get("xgb_spray_mode"),
        "image_classification":  sample.get("image_classification"),
        "manual_classification": sample.get("manual_classification"),   # usually absent
        "video_file":            sample.get("video_file"),              # usually absent
    }

    return features


def process_json_file(
    filepath: Path,
    cursor: sqlite3.Cursor,
    b_coef: np.ndarray,
    a_coef: np.ndarray,
    proc: ElectrosprayDataProcessing,
    npy_dir: Path,
):
    """Load one JSON file and insert all its samples into the database.

    For every sample that carries a raw 'current' array the array is saved to
    ``<npy_dir>/<json_stem>_<sample_key>.npy`` and the path is stored in the
    ``raw_data_file`` column.  Samples without raw data get NULL.
    """
    log.info("Processing  %s", filepath.name)

    with open(filepath, encoding="utf-8") as fh:
        data = json.load(fh)

    solution_name, hv_position = parse_filename(filepath)
    inserted = 0
    skipped  = 0

    for key, sample in data.items():
        if key == "_meta" or not isinstance(sample, dict):
            continue

        try:
            features = process_sample(sample, b_coef, a_coef, proc)
        except Exception as exc:
            log.warning("  ✗ %s / %s – processing error: %s", filepath.name, key, exc)
            skipped += 1
            continue

        # ── Save raw current array to .npy ──────────────────────────────────
        raw_current = sample.get("current")
        if isinstance(raw_current, (list, np.ndarray)) and len(raw_current) > 0:
            safe_key  = re.sub(r"[^\w\-]", "_", key)
            npy_name  = f"{filepath.stem}_{safe_key}.npy"
            npy_path  = npy_dir / npy_name
            np.save(npy_path, np.asarray(raw_current, dtype=np.float32))
            raw_data_file = str(npy_path)
        else:
            raw_data_file = None

        row = (
            features["timestamp"],
            solution_name,
            hv_position,
            features["target_voltage"],
            features["actual_voltage"],
            features["actual_current_ps"],
            features["flow_rate"],
            features["mean_na"],
            features["deviation_na"],
            features["median_na"],
            features["rms_na"],
            features["variance_na"],
            features["qty_max"],
            features["pct_max"],
            features["band_power_v_low"],
            features["band_power_low"],
            features["band_power_mid"],
            features["band_power_high"],
            features["band_power_v_high"],
            features["rf_spray_mode"],
            features["xgb_spray_mode"],
            features["image_classification"],
            features["manual_classification"],
            features["video_file"],
            raw_data_file,
        )

        cursor.execute(_INSERT, row)
        inserted += 1

    log.info("  ✓ %d inserted, %d skipped", inserted, skipped)
    return inserted, skipped


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert Electrospray JSON experiment files to a SQLite database."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing JSON files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--db", "-d",
        default=DEFAULT_DB_PATH,
        help=f"Output SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=FILTER_CUTOFF_HZ,
        help=f"Low-pass filter cutoff in Hz (default: {FILTER_CUTOFF_HZ})",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for input files (default: *.json)",
    )
    parser.add_argument(
        "--npy-dir", "-n",
        default=None,
        help=(
            "Directory where per-sample .npy current arrays are saved. "
            "Defaults to a 'npy/' sub-folder next to the database file. "
            "Pass an absolute path to store them elsewhere."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        raise SystemExit(1)

    json_files = sorted(input_dir.glob(args.pattern))
    if not json_files:
        log.warning("No files matched '%s' in %s", args.pattern, input_dir)
        raise SystemExit(0)

    # Resolve .npy output directory (default: npy/ next to the database)
    npy_dir = Path(args.npy_dir) if args.npy_dir else Path(args.db).parent / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)

    log.info("Found %d JSON file(s) in %s", len(json_files), input_dir)
    log.info("Output database : %s", args.db)
    log.info("Filter cut-off  : %g Hz", args.cutoff)
    log.info("NPY output dir  : %s", npy_dir.resolve())

    # Build filter once; reuse for every sample
    b_coef, a_coef = build_filter_coefficients(args.cutoff)

    # Single processing object; _clear() is called per sample
    proc = ElectrosprayDataProcessing(sample_rate=SAMPLE_RATE)

    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.executescript(_CREATE)
    con.commit()

    total_inserted = 0
    total_skipped  = 0

    try:
        for filepath in json_files:
            ins, skp = process_json_file(filepath, cur, b_coef, a_coef, proc, npy_dir)
            total_inserted += ins
            total_skipped  += skp
            con.commit()   # commit after each file for crash-safety
    finally:
        con.close()

    log.info("─" * 60)
    log.info("Done. Total inserted: %d  |  skipped: %d", total_inserted, total_skipped)
    log.info("Database saved to: %s", Path(args.db).resolve())


if __name__ == "__main__":
    main()