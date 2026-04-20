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
                        trigger_fn: Optional[Callable] = None,
                        temperature: float = 0.0,
                        humidity:    float = 0.0) -> dict:
    """
    Blocking call — returns when oscilloscope has a full 0.5 s record.

    Parameters
    ----------
    trigger_fn : callable with no arguments, or None.
                 Called immediately after scp.start() so the Arduino
                 shutter trigger fires in sync with oscilloscope acquisition.
                 Pass camera._trigger  (the bound method, not camera._trigger())
                 Example in main_electrospray.py:
                     result = acquire_and_process(..., trigger_fn=camera._trigger)
    """

    # ── Acquire + trigger ─────────────────────────────────────────────
    scp.start()

    # Fire the trigger immediately after scp.start() — no delay,
    # no thread handoff, minimum possible latency.
    if trigger_fn is not None:
        try:
            trigger_fn()
        except Exception as e:
            print(f"[ACQ] Trigger error: {e}")

    while not scp.is_data_ready:
        time.sleep(0.01)

    raw        = scp.get_data()
    timestamp  = datetime.now()
    datapoints = np.array(raw[1]) * MULTIPLIER_NA   # [nA]

    # ── Filter ────────────────────────────────────────────────────────
    cutoff = CUTOFF_HZ / (0.5 * SAMPLING_FREQ)
    b, a   = butter(6, Wn=cutoff, btype="low", analog=False)

    # ── Signal processing ─────────────────────────────────────────────
    processing.calculate_filter(a, b, datapoints)
    processing.calculate_fft_raw(datapoints)
    processing.calculate_statistics(datapoints)
    processing.calculate_power_spectral_density(datapoints)

    max_data, qty_max, pct_max = processing.calculate_peaks_signal(datapoints)
    fft_peaks, n_fft_peaks     = processing.calculate_peaks_fft(datapoints)

    processing.calculate_fft_filtered()
    processing.calculate_fft_peaks()

    return {
        "datapoints":          datapoints,
        "timestamp":           timestamp,
        "target_voltage":      target_voltage,
        "actual_voltage":      actual_voltage,
        "actual_current_ps":   actual_current_ps,
        "flow_rate":           float(flow_rate),
        "temperature":         temperature,
        "humidity":            humidity,
        "mean":                processing.mean_value,
        "std":                 processing.stddev,
        "median":              processing.med,
        "rms":                 processing.rms,
        "variance":            processing.variance,
        "rf_classification":   "N/A",   # filled by classify_sample()
        "xgb_classification":  "N/A",
    }