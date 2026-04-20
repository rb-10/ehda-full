"""
Live dashboard  –  must run in the main thread (matplotlib limitation).

Layout
──────
  ┌─────────────────────────┬─────────────────────────┐
  │   Raw current           │   STATUS PANEL          │
  ├─────────────────────────│   V / Q / I stats       │
  │   LP-filtered current   │   Classifications       │
  ├─────────────────────────├─────────────────────────┤
  │   FFT magnitude         │   SPRAY MODE MAP        │
  │                         │   (V vs Q, coloured)    │
  └─────────────────────────┴─────────────────────────┘

Result dict keys used:
  actual_voltage, flow_rate, mean, std, median, rms  — operating stats
  rf_classification   — Random Forest prediction string e.g. "cone_jet  (91%)"
  xgb_classification  — XGBoost prediction string      e.g. "cone_jet  (88%)"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


SAMPLING_FREQ = 1e5
N_PTS         = 50_000

# Canonical mode labels (lowercase, matching spray_mode field in JSON)
MODE_COLORS = {
    "cone_jet":     "#228833",
    "dripping":     "#4477AA",
    "intermittent": "#CCBB44",
    "multi_jet":    "#EE6677",
    "corona":       "#AA3377",
    "undefined":    "#BBBBBB",
    "N/A":          "#DDDDDD",
}

# Map confidence thresholds to visual cues
HIGH_CONF = 0.80   # above this → use full mode color
LOW_CONF  = 0.50   # below this → show as gray (uncertain)


def _parse_prediction(pred_str: str) -> tuple:
    """
    Parse a prediction string like "cone_jet  (91%)" into (mode, confidence).
    Returns (pred_str, 1.0) if format is unexpected.
    """
    if not pred_str or pred_str == "N/A":
        return "N/A", 0.0
    try:
        if "(" in pred_str:
            mode = pred_str.split("(")[0].strip()
            conf = float(pred_str.split("(")[1].replace("%)", "").strip()) / 100
            return mode, conf
    except Exception:
        pass
    return pred_str.strip(), 1.0


def _mode_color(mode: str, confidence: float) -> str:
    """Return display color, grayed out if confidence is low."""
    if confidence < LOW_CONF:
        return "#BBBBBB"
    return MODE_COLORS.get(mode, "#BBBBBB")


class Dashboard:

    def __init__(self):
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.suptitle("Electrospray Mapping – Live",
                          fontsize=12, fontweight="bold")

        gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1],
                               hspace=0.55, wspace=0.35)

        self.ax_raw  = self.fig.add_subplot(gs[0, 0])
        self.ax_filt = self.fig.add_subplot(gs[1, 0])
        self.ax_fft  = self.fig.add_subplot(gs[2, 0])
        self.ax_info = self.fig.add_subplot(gs[0:2, 1])
        self.ax_map  = self.fig.add_subplot(gs[2, 1])

        self.ax_raw.set(xlabel="Time [s]",       ylabel="Current [nA]",
                        title="Raw signal",       ylim=[-300, 1500])
        self.ax_filt.set(xlabel="Time [s]",      ylabel="Current [nA]",
                         title="LP-filtered (3 kHz cut-off)", ylim=[-10, 1000])
        self.ax_fft.set(xlabel="Frequency [Hz]", ylabel="Magnitude",
                        title="FFT",              ylim=[0, 1e6])
        self.ax_info.axis("off")
        self.ax_map.set(xlabel="Voltage [V]",    ylabel="Flow rate [µL/min]",
                        title="Spray mode map")

        self._t = np.arange(N_PTS) / SAMPLING_FREQ
        self._f = np.fft.rfftfreq(N_PTS, d=1 / SAMPLING_FREQ)

        (self.ln_raw,)  = self.ax_raw.plot(
            self._t, np.zeros(N_PTS), lw=0.5, color="#4477AA")
        (self.ln_filt,) = self.ax_filt.plot(
            self._t, np.zeros(N_PTS), lw=0.8, color="#228833")
        (self.ln_fft,)  = self.ax_fft.plot(
            self._f, np.zeros(len(self._f)), lw=0.7, color="#EE6677")

        plt.show(block=False)
        plt.pause(0.1)

        # list of (voltage, flow_rate, mode, confidence) for the map
        self._map_points = []

    # ── Update ────────────────────────────────────────────────────────

    def update(self, result: dict, processing, voltage: float, flow_rate):
        """Refresh all panels with the latest result. Call from main thread."""

        data = result["datapoints"]
        n    = len(data)

        t = self._t[:n] if n <= N_PTS else np.arange(n) / SAMPLING_FREQ
        f = np.fft.rfftfreq(n, d=1 / SAMPLING_FREQ)

        # ── Signal plots ──────────────────────────────────────────────
        self.ln_raw.set_data(t, data)
        self.ax_raw.relim()
        self.ax_raw.autoscale_view(scalex=False)

        filt = np.asarray(processing.datapoints_filtered)
        self.ln_filt.set_data(t, filt[:n] if len(filt) >= n else filt)
        self.ax_filt.relim()
        self.ax_filt.autoscale_view(scalex=False)

        fft_mag = np.abs(np.fft.rfft(data))
        self.ln_fft.set_data(f, fft_mag)
        self.ax_fft.relim()
        self.ax_fft.autoscale_view(scalex=False)

        # ── Parse predictions ─────────────────────────────────────────
        rf_str  = result.get("rf_classification",  "N/A")
        xgb_str = result.get("xgb_classification", "N/A")

        rf_mode,  rf_conf  = _parse_prediction(rf_str)
        xgb_mode, xgb_conf = _parse_prediction(xgb_str)

        # Use RF as the primary mode for the map color
        # (falls back to XGB if RF is N/A)
        primary_mode = rf_mode if rf_mode != "N/A" else xgb_mode
        primary_conf = rf_conf if rf_mode != "N/A" else xgb_conf
        map_color    = _mode_color(primary_mode, primary_conf)

        # ── Status panel ──────────────────────────────────────────────
        self.ax_info.clear()
        self.ax_info.axis("off")

        rows = [
            ("── Operating point ──────────", "black",       9,  "bold"),
            (f"  Voltage    {result['actual_voltage']:>8.0f} V",   "black", 9, "normal"),
            (f"  Flow rate  {float(flow_rate):>8.1f} µL/min",      "black", 9, "normal"),
            (f"  I mean     {result['mean']:>8.1f} nA",             "black", 9, "normal"),
            (f"  I std      {result['std']:>8.1f} nA",              "black", 9, "normal"),
            (f"  I median   {result['median']:>8.1f} nA",           "black", 9, "normal"),
            (f"  I rms      {result['rms']:>8.1f} nA",              "black", 9, "normal"),
            ("",                                                    "black",  5, "normal"),
            ("── Classification ───────────", "black",       9,  "bold"),
            (f"  RF   →  {rf_str}",
             _mode_color(rf_mode, rf_conf),   10, "bold"),
            (f"  XGB  →  {xgb_str}",
             _mode_color(xgb_mode, xgb_conf),  9, "normal"),
        ]

        for i, (txt, clr, sz, wt) in enumerate(rows):
            self.ax_info.text(0.03, 0.97 - i * 0.083, txt,
                              transform=self.ax_info.transAxes,
                              fontsize=sz, color=clr, fontweight=wt,
                              va="top", family="monospace")

        # ── Spray mode map ────────────────────────────────────────────
        self._map_points.append(
            (float(voltage), float(flow_rate), primary_mode, primary_conf)
        )

        self.ax_map.clear()
        self.ax_map.set(xlabel="Voltage [V]", ylabel="Flow rate [µL/min]",
                        title="Spray mode map")

        xs   = [p[0] for p in self._map_points]
        ys   = [p[1] for p in self._map_points]
        cols = [_mode_color(p[2], p[3]) for p in self._map_points]
        self.ax_map.scatter(xs, ys, c=cols, s=80, zorder=3,
                            edgecolors="white", linewidths=0.4)

        legend = [Patch(facecolor=c, label=m)
                  for m, c in MODE_COLORS.items() if m != "N/A"]
        self.ax_map.legend(handles=legend, fontsize=6,
                           loc="upper left", framealpha=0.8)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()