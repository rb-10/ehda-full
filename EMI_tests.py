from mapping.software.database            import ElectrosprayDatabase
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import os
import sys

SAMPLE_RATE = 100_000  # Hz
NPY_DIR = "data/raw_waveforms"
BASE = Path(r"C:\Users\HV\Desktop\bruno_work\main\data")
db = ElectrosprayDatabase(str(BASE))
    

# 1. Load Data
df_db = db.load_training_dataframe()
print(f"✓ Loaded {len(df_db)} records from database")
EXCLUDE_FEATURES_MODIFIED = [
    "actual_current_ps",
    "current_PS",
    "voltage_error",
    "target_voltage",
    "variance_na", "rms_na", "band_power_v_low", "band_power_low", "band_power_mid", "band_power_high", "band_power_v_high", "peak", "crest_factor",
    "kurtosis", "skewness", "peak_to_peak", "zero_crossing_rate", "dominant_freq", "mean_freq", "spectral_entropy", "total_power", "wt_approx_L6_energy",
    "wt_approx_L6_energy_rel", "wt_detail_L6_energy", "wt_detail_L6_energy_rel", "wt_detail_L5_energy", "wt_detail_L5_energy_rel", "wt_detail_L4_energy",
    "wt_detail_L4_energy_rel", "wt_detail_L3_energy", "wt_detail_L3_energy_rel", "wt_detail_L2_energy", "wt_detail_L2_energy_rel", "wt_detail_L1_energy",
    "wt_detail_L1_energy_rel", "deviation_na", "median_na", "qty_max", "pct_max"
    # Add more features here if you want to exclude them
    # Example: "some_feature_name",
]
df_db = df_db.drop(
    columns=[c for c in EXCLUDE_FEATURES_MODIFIED if c in df_db.columns]
)

def load_waveform(filename: str) -> np.ndarray | None:
    path = os.path.join(NPY_DIR, filename)
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    return arr.flatten()
 
 
def format_classification(val):
    return str(val) if val not in (None, "", "N/A", float("nan")) else "—"


class WaveformViewer:
    # Colour palette
    BG        = "#0d1117"
    PANEL_BG  = "#161b22"
    ACCENT    = "#58a6ff"
    VOLT_CLR  = "#f0a500"
    FLOW_CLR  = "#3fb950"
    WAVE_CLR  = "#58a6ff"
    GRID_CLR  = "#21262d"
    TEXT_CLR  = "#e6edf3"
    MUTED_CLR = "#8b949e"
    WARN_CLR  = "#f85149"
 
    def __init__(self, dataframe: pd.DataFrame):
        self.df    = dataframe
        self.index = 0
        self.total = len(dataframe)
        self._build_figure()
        self._render()
 
    # ── Figure scaffold ──────────────────────────────────────────────────────
 
    def _build_figure(self):
        plt.rcParams.update({
            "figure.facecolor":  self.BG,
            "axes.facecolor":    self.PANEL_BG,
            "axes.edgecolor":    self.GRID_CLR,
            "axes.labelcolor":   self.TEXT_CLR,
            "xtick.color":       self.MUTED_CLR,
            "ytick.color":       self.MUTED_CLR,
            "grid.color":        self.GRID_CLR,
            "text.color":        self.TEXT_CLR,
            "font.family":       "monospace",
        })
 
        self.fig = plt.figure(figsize=(14, 8), facecolor=self.BG)
        self.fig.canvas.manager.set_window_title("Waveform Viewer")
 

        gs = gridspec.GridSpec(
            3, 1,
            figure=self.fig,
            height_ratios=[0.18, 0.14, 1],
            hspace=0.06,
            left=0.07, right=0.97, top=0.93, bottom=0.09,
        )

        # ── Title bar (text-only axis) ──
        self.ax_title = self.fig.add_subplot(gs[0])
        self.ax_title.set_axis_off()

        # ── Metadata bar ──
        self.ax_meta = self.fig.add_subplot(gs[1])
        self.ax_meta.set_axis_off()

        # ── Waveform ──
        self.ax_wave = self.fig.add_subplot(gs[2])
        self.ax_wave.grid(True, linewidth=0.4, alpha=0.6)
        self.ax_wave.set_xlabel("Time (ms)", fontsize=9)
        self.ax_wave.set_ylabel("Current (nA)", fontsize=9)

        # ── Zoom inset (bottom-right, created once) ──
        self.ax_zoom = self.fig.add_axes([0.72, 0.10, 0.24, 0.22])  # [left, bottom, width, height]
        self.ax_zoom.set_facecolor(self.PANEL_BG)
        for spine in self.ax_zoom.spines.values():
            spine.set_edgecolor(self.ACCENT)
            spine.set_linewidth(1.2)
 
 
        # Navigation hint at bottom
        self.fig.text(
            0.5, 0.01,
            "← / p  previous     → / n  next     q  quit",
            ha="center", va="bottom",
            fontsize=8, color=self.MUTED_CLR,
        )
 
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
 
    # ── Rendering ────────────────────────────────────────────────────────────
 
    def _render(self):
        row = self.df.iloc[self.index]
 
        # ── Title bar ──────────────────────────────────────────────────────
        self.ax_title.clear()
        self.ax_title.set_axis_off()
 
        # Progress pill
        progress_txt = f"  {self.index + 1} / {self.total}  "
        self.ax_title.text(
            0.0, 0.5, progress_txt,
            transform=self.ax_title.transAxes,
            fontsize=9, color=self.BG,
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=self.ACCENT, edgecolor="none"),
        )
 
        # Solution name
        self.ax_title.text(
            0.08, 0.5, row["solution_name"],
            transform=self.ax_title.transAxes,
            fontsize=14, fontweight="bold",
            color=self.TEXT_CLR, va="center",
        )
 
        # Timestamp (right-aligned)
        self.ax_title.text(
            1.0, 0.5, row["timestamp"],
            transform=self.ax_title.transAxes,
            fontsize=8, color=self.MUTED_CLR,
            va="center", ha="right",
        )
 
        # ── Metadata bar ───────────────────────────────────────────────────
        self.ax_meta.clear()
        self.ax_meta.set_axis_off()
 
        meta_items = [
            ("VOLTAGE",   f"{row['actual_voltage']:.2f} V",   self.VOLT_CLR),
            ("FLOW RATE", f"{row['flow_rate']:.2f} µL/min",   self.FLOW_CLR),
            ("HV POS",    str(row["hv_position"]),             self.ACCENT),
            ("MEAN nA",   f"{row['mean_na']:.3f}",             self.TEXT_CLR),
            ("RF MODE",   format_classification(row["rf_spray_mode"]),  self.MUTED_CLR),
            ("XGB MODE",  format_classification(row["xgb_spray_mode"]), self.MUTED_CLR),
            ("IMG CLASS", format_classification(row["image_classification"]), self.MUTED_CLR),
            ("MANUAL",    format_classification(row["manual_classification"]), self.MUTED_CLR),
        ]
 
        x = 0.0
        for label, value, colour in meta_items:
            self.ax_meta.text(
                x, 0.80, label,
                transform=self.ax_meta.transAxes,
                fontsize=6.5, color=self.MUTED_CLR, va="top",
            )
            self.ax_meta.text(
                x, 0.35, value,
                transform=self.ax_meta.transAxes,
                fontsize=8.5, color=colour, va="top", fontweight="bold",
            )
            x += 0.125
 
        # ── Waveform ───────────────────────────────────────────────────────
        self.ax_wave.clear()
        self.ax_wave.set_facecolor(self.PANEL_BG)
        self.ax_wave.grid(True, linewidth=0.4, alpha=0.5, color=self.GRID_CLR)
        self.ax_wave.set_xlabel("Time (s)", fontsize=9, color=self.MUTED_CLR)
        self.ax_wave.set_ylabel("Current (nA)", fontsize=9, color=self.MUTED_CLR)
        self.ax_wave.tick_params(labelsize=8)
 
        # Voltage & flow rate horizontal reference lines
        v_norm = row["actual_voltage"]
        fr_norm = row["flow_rate"]
 
        waveform = load_waveform(row["raw_data_file"])
 
        if waveform is not None:
            n = len(waveform)
            t = np.linspace(0, n / SAMPLE_RATE, n)
            self.ax_wave.plot(t, waveform, color=self.WAVE_CLR, linewidth=0.5, alpha=0.9)
 
            # Overlay mean line
            mean_val = np.mean(waveform)
            self.ax_wave.axhline(
                mean_val, color=self.ACCENT, linewidth=1.0,
                linestyle="--", alpha=0.7,
                label=f"mean = {mean_val:.2f} nA",
            )
 
            self.ax_wave.set_xlim(t[0], t[-1])
            self.ax_wave.set_title(
                row["raw_data_file"],
                fontsize=8, color=self.MUTED_CLR, pad=4,
            )
            self.ax_wave.legend(
                fontsize=8, facecolor=self.PANEL_BG,
                edgecolor=self.GRID_CLR, labelcolor=self.TEXT_CLR,
                loc="upper right",
            )
        else:
            # File not found — show a clear warning
            self.ax_wave.text(
                0.5, 0.5,
                f"⚠  File not found:\n{row['raw_data_file']}",
                transform=self.ax_wave.transAxes,
                fontsize=11, color=self.WARN_CLR,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.6", facecolor=self.PANEL_BG, edgecolor=self.WARN_CLR),
            )
        # ── Zoom panel ─────────────────────────────────────────────────────────
        self.ax_zoom.clear()
        self.ax_zoom.set_facecolor(self.PANEL_BG)
        for spine in self.ax_zoom.spines.values():
            spine.set_edgecolor(self.ACCENT)
            spine.set_linewidth(1.2)
        
        if waveform is not None:
            n = len(waveform)
            # 1% of samples centred on the middle
            half_window = int(n * 0.005)          # half of 1%
            mid = n // 2
            z_start = mid - half_window
            z_end   = mid + half_window
        
            z_wave = waveform[z_start:z_end]
            z_t    = t[z_start:z_end]             # reuse the ms time array from above
        
            self.ax_zoom.plot(z_t, z_wave, color=self.WAVE_CLR, linewidth=0.8)
            self.ax_zoom.axhline(np.mean(z_wave), color=self.ACCENT,
                                 linewidth=0.8, linestyle="--", alpha=0.7)
            self.ax_zoom.set_xlim(z_t[0], z_t[-1])
            self.ax_zoom.tick_params(labelsize=7, colors=self.MUTED_CLR)
            self.ax_zoom.set_xlabel("Time (ms)", fontsize=7, color=self.MUTED_CLR)
            self.ax_zoom.set_ylabel("nA",        fontsize=7, color=self.MUTED_CLR)
            self.ax_zoom.grid(True, linewidth=0.3, alpha=0.4, color=self.GRID_CLR)
            self.ax_zoom.set_title("zoom  ×100  (centre 1%)",
                                    fontsize=7, color=self.ACCENT, pad=3)
        
            # Draw a subtle highlight span on the main plot
            self.ax_wave.axvspan(z_t[0], z_t[-1],
                                 alpha=0.08, color=self.ACCENT, zorder=0)
        else:
            self.ax_zoom.set_axis_off()
        # Voltage annotation (top-right corner of wave panel)
        self.ax_wave.annotate(
            f"⚡ {v_norm:.1f} V",
            xy=(0.01, 0.97), xycoords="axes fraction",
            fontsize=9, color=self.VOLT_CLR, va="top",
            fontweight="bold",
        )
        self.ax_wave.annotate(
            f"💧 {fr_norm:.2f} µL/min",
            xy=(0.01, 0.88), xycoords="axes fraction",
            fontsize=9, color=self.FLOW_CLR, va="top",
            fontweight="bold",
        )
 
        self.fig.canvas.draw_idle()
 
    # ── Key handler ──────────────────────────────────────────────────────────
 
    def _on_key(self, event):
        if event.key in ("right", "n"):
            if self.index < self.total - 1:
                self.index += 1
                self._render()
            else:
                print("Already at last record.")
        elif event.key in ("left", "p"):
            if self.index > 0:
                self.index -= 1
                self._render()
            else:
                print("Already at first record.")
        elif event.key == "q":
            plt.close("all")
            sys.exit(0)
 
    def show(self):
        plt.show()


viewer = WaveformViewer(df_db)
viewer.show()