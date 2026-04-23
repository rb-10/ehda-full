"""
Single-shot measurement: oscilloscope acquisition → signal processing.

Classification is handled in main_electrospray.py after this returns.
This function only acquires and computes signal statistics.

Result dict keys for classification (set to "N/A" here, filled in main):
  rf_classification   — Random Forest prediction
  xgb_classification  — XGBoost prediction

Trigger:
  An optional trigger callable can be passed in.
  It is called immediately after scp.start() so the Arduino fires
  in sync with the first oscilloscope sample.
"""

import time
import numpy as np
from datetime import datetime
from scipy.signal import butter
from typing import Callable, Optional


SAMPLING_FREQ = 1e5
RECORD_LENGTH = 50_000
MULTIPLIER_NA = 500
CUTOFF_HZ     = 3_000


def acquire_and_process(scp,
                        target_voltage: float,
                        flow_rate,
                        actual_voltage: float,
                        actual_current_ps: float,
                        processing,
                        trigger_fn: Optional[Callable] = None) -> dict:
    """
    Acquires 0.5s of data, processes features via ElectrosprayDataProcessing,
    and returns a results dictionary including statistics and spectral bands.
    """

    # 1. Reset state to ensure we aren't using peaks from the previous sample
    processing.clear_results()

    # ── Acquire + trigger ─────────────────────────────────────────────
    scp.start()

    if trigger_fn is not None:
        try:
            trigger_fn()
        except Exception as e:
            print(f"[ACQ] Trigger error: {e}")

    while not scp.is_data_ready:
        time.sleep(0.01)

    raw = scp.get_data()
    timestamp = datetime.now()
    datapoints = np.array(raw[1]) * MULTIPLIER_NA   # [nA]

    # ── Filter Design ─────────────────────────────────────────────────
    # We keep this here if CUTOFF_HZ or SAMPLING_FREQ are global/config vars
    cutoff = CUTOFF_HZ / (0.5 * SAMPLING_FREQ)
    b, a = butter(6, Wn=cutoff, btype="low", analog=False)

    # ── Signal processing ─────────────────────────────────────────────
    # A. Apply Filter
    processing.calculate_filter(a, b, datapoints)
    
    # B. Time Domain (Stats on filtered data, Peaks on raw to check clipping)
    processing.calculate_statistics(processing.datapoints_filtered)
    max_val, qty_max, pct_max = processing.calculate_peaks_signal(datapoints)
    
    # C. Frequency Domain (Always use raw data to see full spectrum)
    processing.calculate_fft_raw(processing.datapoints_filtered)
    processing.calculate_power_spectral_density(processing.datapoints_filtered)
    
    # D. Peak finding in FFT
    fft_peaks, n_fft_peaks = processing.calculate_fft_peaks()

    # ── Build Results ─────────────────────────────────────────────────
    # Get the dictionary containing mean, std, and the new band_power features
    stats = processing.get_statistics_dictionary()

    # Create the final return object
    results = {
        "datapoints":        datapoints,
        "timestamp":         timestamp,
        "target_voltage":    target_voltage,
        "actual_voltage":    actual_voltage,
        "actual_current_ps": actual_current_ps,
        "flow_rate":         float(flow_rate),
        "qty_max":           qty_max,  # Useful for detecting saturated sensors
        "pct_max":           pct_max,
        "n_fft_peaks":       n_fft_peaks,
        "rf_classification": "N/A",
        "xgb_classification": "N/A",
    }

    # Merge the statistics and band powers into the main results
    results.update(stats)

    return results