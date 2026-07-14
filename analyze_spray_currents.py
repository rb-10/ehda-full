"""
analyze_spray_currents.py

Interactive Python GUI application using Tkinter and Matplotlib to:
1. Load electrospray experimental runs from data.db / DMF/data.db.
2. Group measurements into 5-sample voltage steps.
3. Calculate average current, mode-to-mode ratios, and identify spray modes.
4. Visualize the I-V curve (Current vs Voltage) with classification color-coding.
5. Highlight the current step and display either its raw current waveforms or sample variance.
6. Provide hypothesis verification comparing average current in cone jet vs multi-jet modes.
7. Support keyboard navigation (n = Next, p = Previous) and mouse clicks on the plot.

Usage:
    .venv\\Scripts\\python.exe analyze_spray_currents.py
"""

import os
import re
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Set matplotlib style for dark mode
plt.style.use('dark_background')

# Settings
MULTIPLIER = 0.002  # Multiplier for raw waveforms from TiePie scope (e.g. to scale correctly)
CLASSES = {
    "cone_jet": {"color": "#2ecc71", "label": "Cone Jet"},
    "multi_jet": {"color": "#e74c3c", "label": "Multi-Jet"},
    "dripping": {"color": "#3498db", "label": "Dripping"},
    "intermitent": {"color": "#f1c40f", "label": "Intermittent"},
    "unconclusive": {"color": "#95a5a6", "label": "Unconclusive"},
    "n/a": {"color": "#7f8c8d", "label": "N/A"}
}

def parse_classification(val):
    """Extract clean label and confidence percentage from database field."""
    if not val or val == 'N/A':
        return 'N/A', None
    match = re.match(r'^([a-zA-Z0-9_\s\-]+)(?:\s*\((\d+%)\))?$', val)
    if match:
        label = match.group(1).strip().lower()
        confidence = match.group(2)
        return label, confidence
    return val.strip().lower(), None

class DatabaseManager:
    """Helper to query database runs and measurements."""
    def __init__(self, db_path):
        self.db_path = db_path
        
    def get_available_runs(self):
        """Returns list of unique (solution_name, video_file) runs with row counts."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT solution_name, video_file, COUNT(*) "
                "FROM measurements "
                "GROUP BY solution_name, video_file"
            )
            runs = []
            for solution, video, count in cursor.fetchall():
                display_name = f"{solution} | {video if video else 'No Video'} ({count} rows)"
                runs.append({
                    'solution': solution,
                    'video': video,
                    'count': count,
                    'display': display_name
                })
            conn.close()
            return runs
        except Exception as e:
            print(f"Error querying runs: {e}")
            return []

    def get_run_steps(self, solution, video):
        """Loads and processes measurements for a run, grouping into 5-sample steps."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if video is None:
                cursor.execute(
                    "SELECT * FROM measurements "
                    "WHERE solution_name = ? AND video_file IS NULL "
                    "ORDER BY timestamp ASC", (solution,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM measurements "
                    "WHERE solution_name = ? AND video_file = ? "
                    "ORDER BY timestamp ASC", (solution, video)
                )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            steps = []
            # Group into 5-sample steps
            for i in range(0, len(rows), 5):
                step_rows = rows[i:i+5]
                if not step_rows:
                    continue
                
                # Compute stats
                mean_current = np.mean([r['mean_na'] for r in step_rows])
                avg_actual_voltage = np.mean([r['actual_voltage'] for r in step_rows])
                target_voltage = step_rows[0]['target_voltage']
                flow_rate = step_rows[0]['flow_rate']
                
                # Image Classification mode (majority voting of non-N/A, fallback to first)
                img_cls_list = [r['image_classification'] for r in step_rows if r['image_classification'] and r['image_classification'] != 'N/A']
                if img_cls_list:
                    img_cls_raw = Counter(img_cls_list).most_common(1)[0][0]
                else:
                    img_cls_raw = step_rows[0]['image_classification'] or 'N/A'
                
                # Manual Classification mode
                man_cls_list = [r['manual_classification'] for r in step_rows if r['manual_classification'] and r['manual_classification'] != 'N/A']
                if man_cls_list:
                    man_cls_raw = Counter(man_cls_list).most_common(1)[0][0]
                else:
                    man_cls_raw = step_rows[0]['manual_classification'] or 'N/A'
                
                # ML Classifications (RF & XGB)
                rf_cls_list = [r['rf_spray_mode'] for r in step_rows if r['rf_spray_mode'] and r['rf_spray_mode'] != 'N/A']
                rf_cls_raw = Counter(rf_cls_list).most_common(1)[0][0] if rf_cls_list else step_rows[0]['rf_spray_mode'] or 'N/A'
                
                xgb_cls_list = [r['xgb_spray_mode'] for r in step_rows if r['xgb_spray_mode'] and r['xgb_spray_mode'] != 'N/A']
                xgb_cls_raw = Counter(xgb_cls_list).most_common(1)[0][0] if xgb_cls_list else step_rows[0]['xgb_spray_mode'] or 'N/A'
                
                steps.append({
                    'step_idx': len(steps) + 1,
                    'solution_name': solution,
                    'flow_rate': flow_rate,
                    'target_voltage': target_voltage,
                    'actual_voltage': avg_actual_voltage,
                    'mean_current': mean_current,
                    'image_classification': img_cls_raw,
                    'manual_classification': man_cls_raw,
                    'rf_spray_mode': rf_cls_raw,
                    'xgb_spray_mode': xgb_cls_raw,
                    'rows': step_rows
                })
            
            # Compute ratios compared to previous steps
            for idx, step in enumerate(steps):
                if idx == 0:
                    step['ratio'] = 1.0
                else:
                    prev_current = steps[idx-1]['mean_current']
                    if prev_current != 0:
                        step['ratio'] = step['mean_current'] / prev_current
                    else:
                        step['ratio'] = 1.0
                        
            return steps
        except Exception as e:
            print(f"Error loading run steps: {e}")
            import traceback; traceback.print_exc()
            return []

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EHDA Spray Mode & Current Variations Analyzer")
        self.geometry("1300x850")
        self.configure(bg="#121212")
        
        # Application state
        self.db_path = None
        self.db_mgr = None
        self.runs = []
        self.steps = []
        self.current_step_idx = 0
        
        # Configure fonts and Tkinter dark styling
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure(".", bg="#121212", fg="#ffffff", fieldbackground="#1e1e1e", font=("Segoe UI", 10))
        self.style.configure("TLabel", background="#121212", foreground="#ffffff")
        self.style.configure("TFrame", background="#121212")
        self.style.configure("TLabelframe", background="#1e1e1e", foreground="#ffffff", bordercolor="#333333")
        self.style.configure("TLabelframe.Label", background="#1e1e1e", foreground="#2e62d4", font=("Segoe UI", 10, "bold"))
        self.style.configure("TButton", background="#2e62d4", foreground="#ffffff", borderwidth=0, font=("Segoe UI", 10, "bold"))
        self.style.map("TButton", background=[("active", "#4a7cf5")])
        self.style.configure("Accent.TButton", background="#e74c3c", foreground="#ffffff")
        self.style.map("Accent.TButton", background=[("active", "#ff6b5b")])
        
        # Build UI layout
        self.create_widgets()
        
        # Key bindings
        self.bind("<Key-n>", lambda e: self.next_step())
        self.bind("<Key-p>", lambda e: self.prev_step())
        self.bind("<Key-N>", lambda e: self.next_step())
        self.bind("<Key-P>", lambda e: self.prev_step())
        
        # Attempt to auto-load default databases
        self.auto_load_db()

    def auto_load_db(self):
        """Tries to find database file in data/data.db or DMF/data.db on startup."""
        for path in ["data/data.db", "DMF/data.db", "data.db"]:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                self.load_database(abs_path)
                return

    def load_database(self, path):
        """Loads database, queries runs, updates UI."""
        self.db_path = path
        self.db_entry.delete(0, tk.END)
        self.db_entry.insert(0, path)
        self.db_mgr = DatabaseManager(path)
        
        self.runs = self.db_mgr.get_available_runs()
        if not self.runs:
            messagebox.showwarning("Database Empty", "No measurements found in database.")
            return
            
        # Update dropdown options
        self.run_combo['values'] = [r['display'] for r in self.runs]
        self.run_combo.current(0)
        self.on_run_selected()

    def browse_db(self):
        """Opens file dialog to locate data.db."""
        path = filedialog.askopenfilename(
            title="Select SQLite Database File",
            filetypes=[("Database Files", "*.db"), ("All Files", "*.*")]
        )
        if path:
            self.load_database(path)

    def on_run_selected(self, event=None):
        """Triggered when run dropdown selection changes."""
        idx = self.run_combo.current()
        if idx == -1:
            return
        run = self.runs[idx]
        self.steps = self.db_mgr.get_run_steps(run['solution'], run['video'])
        self.current_step_idx = 0
        
        # Perform overall verification statistics
        self.verify_hypothesis()
        
        # Populate UI and draw plots
        if self.steps:
            self.select_step(0)
        else:
            messagebox.showwarning("Empty Run", "No valid steps could be parsed for this run.")

    def verify_hypothesis(self):
        """Analyzes average currents of cone jet vs multi-jet steps to verify hypothesis."""
        cone_currents = []
        multi_currents = []
        
        for step in self.steps:
            img_mode, _ = parse_classification(step['image_classification'])
            current = step['mean_current']
            if img_mode == 'cone_jet':
                cone_currents.append(current)
            elif img_mode == 'multi_jet':
                multi_currents.append(current)
                
        # Update Verification Box text
        self.hypo_text.configure(state='normal')
        self.hypo_text.delete('1.0', tk.END)
        
        self.hypo_text.insert(tk.END, "HYPOTHESIS VERIFICATION:\n", "title")
        self.hypo_text.insert(tk.END, "Current in multi-jet mode > cone jet mode.\n\n")
        
        if not cone_currents and not multi_currents:
            self.hypo_text.insert(tk.END, "No cone jet or multi-jet steps detected in this run via image classification.\n", "warning")
        elif not cone_currents:
            self.hypo_text.insert(tk.END, f"Multi-Jet steps: {len(multi_currents)} (Avg: {np.mean(multi_currents):.2f} nA)\n")
            self.hypo_text.insert(tk.END, "No Cone Jet steps detected. Cannot compute ratio.\n", "warning")
        elif not multi_currents:
            self.hypo_text.insert(tk.END, f"Cone Jet steps: {len(cone_currents)} (Avg: {np.mean(cone_currents):.2f} nA)\n")
            self.hypo_text.insert(tk.END, "No Multi-Jet steps detected. Cannot compute ratio.\n", "warning")
        else:
            avg_cone = np.mean(cone_currents)
            avg_multi = np.mean(multi_currents)
            ratio = avg_multi / avg_cone
            
            self.hypo_text.insert(tk.END, f"Cone Jet Mode (Avg): {avg_cone:.2f} nA  ({len(cone_currents)} steps)\n")
            self.hypo_text.insert(tk.END, f"Multi-Jet Mode (Avg): {avg_multi:.2f} nA  ({len(multi_currents)} steps)\n")
            self.hypo_text.insert(tk.END, f"Current Ratio (Multi/Cone): {ratio:.2f}x\n\n")
            
            if ratio > 1.2:
                self.hypo_text.insert(tk.END, "STATUS: HYPOTHESIS SUPPORTED!\n", "success")
                self.hypo_text.insert(tk.END, f"The current is significantly higher in multi-jet mode (approx. {ratio:.1f}x increase).", "success")
            elif ratio > 1.0:
                self.hypo_text.insert(tk.END, "STATUS: WEAK SUPPORT\n", "warning")
                self.hypo_text.insert(tk.END, f"Multi-jet current is slightly higher ({ratio:.2f}x).", "warning")
            else:
                self.hypo_text.insert(tk.END, "STATUS: HYPOTHESIS REFUTED\n", "danger")
                self.hypo_text.insert(tk.END, f"Multi-jet current ({avg_multi:.2f} nA) is equal or lower than cone jet ({avg_cone:.2f} nA).", "danger")
                
        self.hypo_text.configure(state='disabled')

    def select_step(self, idx):
        """Selects a specific step index, updates stats display and plots."""
        if not self.steps:
            return
        self.current_step_idx = max(0, min(idx, len(self.steps) - 1))
        step = self.steps[self.current_step_idx]
        
        # Update labels in left panel
        self.lbl_sol.configure(text=f"Solution: {step['solution_name']}")
        self.lbl_flow.configure(text=f"Flow Rate: {step['flow_rate']} µL/min")
        self.lbl_step_num.configure(text=f"Step {self.current_step_idx + 1} of {len(self.steps)}")
        self.lbl_target_v.configure(text=f"Target Voltage: {step['target_voltage']:.0f} V")
        self.lbl_actual_v.configure(text=f"Actual Voltage: {step['actual_voltage']:.1f} V")
        self.lbl_mean_i.configure(text=f"Average Current: {step['mean_current']:.3f} nA")
        
        # Display ratio with color/formatting
        ratio = step['ratio']
        if self.current_step_idx == 0:
            self.lbl_ratio.configure(text="Ratio to Prev: N/A", foreground="#ffffff")
        else:
            self.lbl_ratio.configure(text=f"Ratio to Prev: {ratio:.2f}x")
            if ratio > 1.5:
                self.lbl_ratio.configure(foreground="#ff9100")  # Highlight significant jumps
            else:
                self.lbl_ratio.configure(foreground="#ffffff")

        # Handle classifications labels and coloring
        img_label, img_conf = parse_classification(step['image_classification'])
        man_label, man_conf = parse_classification(step['manual_classification'])
        
        # Get color properties
        img_style = CLASSES.get(img_label, CLASSES["unconclusive"])
        man_style = CLASSES.get(man_label, CLASSES["n/a"])
        
        conf_str = f" ({img_conf})" if img_conf else ""
        self.lbl_img_class.configure(
            text=f"Image Classification:\n  {img_style['label']}{conf_str}",
            foreground=img_style['color']
        )
        self.lbl_man_class.configure(
            text=f"Manual Classification:\n  {man_style['label']}",
            foreground=man_style['color']
        )
        
        # ML models helper
        rf_label, _ = parse_classification(step['rf_spray_mode'])
        xgb_label, _ = parse_classification(step['xgb_spray_mode'])
        self.lbl_rf_class.configure(text=f"RF Spray Mode: {step['rf_spray_mode']}")
        self.lbl_xgb_class.configure(text=f"XGB Spray Mode: {step['xgb_spray_mode']}")
        
        # Redraw plots
        self.update_plots()

    def update_plots(self):
        """Redraws the I-V curve and the raw waveform (or stability bar chart)."""
        if not self.steps:
            return
            
        current_step = self.steps[self.current_step_idx]
        
        # ── SUBPLOT 1: I-V Curve ──
        self.ax1.clear()
        self.ax1.set_title("Experimental I-V Curve (Average Current vs Voltage)", fontsize=11, color="white", fontweight="bold")
        self.ax1.set_xlabel("Actual Voltage (V)", fontsize=9, color="white")
        self.ax1.set_ylabel("Average Current (nA)", fontsize=9, color="white")
        self.ax1.grid(True, linestyle="--", alpha=0.2, color="#777777")
        
        voltages = [s['actual_voltage'] for s in self.steps]
        currents = [s['mean_current'] for s in self.steps]
        
        # Plot baseline trajectory line
        self.ax1.plot(voltages, currents, color="#333333", alpha=0.7, zorder=1, linewidth=1.5)
        
        # Plot points color-coded by detected image mode
        for mode_key, style in CLASSES.items():
            pts_x = []
            pts_y = []
            for s in self.steps:
                lbl, _ = parse_classification(s['image_classification'])
                if lbl == mode_key:
                    pts_x.append(s['actual_voltage'])
                    pts_y.append(s['mean_current'])
            
            if pts_x:
                # Select markers
                marker = 'o'
                if mode_key == 'cone_jet': marker = 'o'
                elif mode_key == 'multi_jet': marker = 's'
                elif mode_key == 'dripping': marker = '^'
                elif mode_key == 'intermitent': marker = 'D'
                
                self.ax1.scatter(
                    pts_x, pts_y, 
                    color=style['color'], 
                    label=style['label'], 
                    s=40, 
                    edgecolor="#222222", 
                    linewidths=0.5,
                    zorder=2
                )
        
        # Highlight current step
        self.ax1.scatter(
            current_step['actual_voltage'], current_step['mean_current'],
            color="#ffffff", s=150, edgecolor="#ff2a2a", linewidths=2.5,
            marker="*", label="Current Step", zorder=3
        )
        self.ax1.legend(loc="upper left", frameon=True, facecolor="#1e1e1e", edgecolor="#333333", fontsize=8)
        
        # ── SUBPLOT 2: Waveforms / Stability ──
        self.ax2.clear()
        
        # Try to locate waveforms
        db_dir = os.path.dirname(self.db_path)
        waveforms_dir = os.path.join(db_dir, "raw_waveforms")
        
        waveforms_loaded = []
        filenames = []
        
        for row in current_step['rows']:
            raw_file = row.get('raw_data_file')
            if raw_file:
                fp = os.path.join(waveforms_dir, raw_file)
                if os.path.exists(fp):
                    try:
                        waveforms_loaded.append(np.load(fp) * MULTIPLIER)
                        filenames.append(raw_file)
                    except Exception as e:
                        print(f"Error loading {raw_file}: {e}")
                        
        if waveforms_loaded:
            # We found the waveform files! Overlay them
            self.ax2.set_title(f"Step Waveforms (5 samples overlaid) - Voltage: {current_step['actual_voltage']:.0f} V", fontsize=11, color="white", fontweight="bold")
            self.ax2.set_xlabel("Oscilloscope Sample Index", fontsize=9, color="white")
            self.ax2.set_ylabel("Signal Voltage (V)", fontsize=9, color="white")
            
            colors = ["#2e62d4", "#00adb5", "#393e46", "#ff9100", "#e74c3c"]
            for idx, wf in enumerate(waveforms_loaded):
                self.ax2.plot(
                    wf, 
                    alpha=0.8 - (idx * 0.08), 
                    color=colors[idx % len(colors)], 
                    linewidth=0.6,
                    label=f"Sample {idx+1}"
                )
            self.ax2.set_ylim(-4.0, 4.0)  # Standard TiePie scale
            self.ax2.grid(True, linestyle="--", alpha=0.2, color="#777777")
            self.ax2.legend(loc="upper right", frameon=True, facecolor="#1e1e1e", edgecolor="#333333", fontsize=8)
        else:
            # Waveforms missing. Plot stability (the 5 sample currents)
            self.ax2.set_title(f"Stability Map: 5 Samples Current Variance (Step {current_step['step_idx']})", fontsize=11, color="white", fontweight="bold")
            self.ax2.set_xlabel("Sample Index", fontsize=9, color="white")
            self.ax2.set_ylabel("Current Value (nA)", fontsize=9, color="white")
            
            sample_currents = [r['mean_na'] for r in current_step['rows']]
            indices = list(range(1, 6))
            
            # Set background style for bar chart
            bars = self.ax2.bar(
                indices, sample_currents, 
                color="#2e62d4", edgecolor="#1b3d87", alpha=0.8, width=0.5
            )
            
            # Highlight variance
            for bar in bars:
                height = bar.get_height()
                self.ax2.text(
                    bar.get_x() + bar.get_width()/2.0, 
                    height + (max(sample_currents)*0.01), 
                    f"{height:.3f}", 
                    ha='center', va='bottom', fontsize=8, color='white'
                )
            
            # Adjust y limits for better visualization
            self.ax2.set_xticks(indices)
            self.ax2.set_xticklabels([f"S{i}" for i in indices])
            margin = max(0.1, max(sample_currents) * 0.15)
            self.ax2.set_ylim(0, max(sample_currents) + margin)
            self.ax2.grid(True, axis='y', linestyle="--", alpha=0.2, color="#777777")
            
            # Subtle text explaining fallback
            self.ax2.text(
                0.98, 0.05, "Waveform files missing. Plotting samples average current.", 
                color="#888888", fontsize=8, ha='right', va='bottom', transform=self.ax2.transAxes
            )
            
        self.canvas.draw()

    def next_step(self):
        """Advance to the next step in the sequence."""
        if self.current_step_idx < len(self.steps) - 1:
            self.select_step(self.current_step_idx + 1)

    def prev_step(self):
        """Go back to the previous step in the sequence."""
        if self.current_step_idx > 0:
            self.select_step(self.current_step_idx - 1)

    def on_plot_clicked(self, event):
        """Finds closest point in the I-V plot and jumps to its step index."""
        if event.inaxes != self.ax1 or not self.steps:
            return
        click_x, click_y = event.xdata, event.ydata
        
        voltages = [s['actual_voltage'] for s in self.steps]
        currents = [s['mean_current'] for s in self.steps]
        
        x_range = max(voltages) - min(voltages) if max(voltages) != min(voltages) else 1.0
        y_range = max(currents) - min(currents) if max(currents) != min(currents) else 1.0
        
        min_dist = float('inf')
        closest_idx = self.current_step_idx
        
        for idx, step in enumerate(self.steps):
            dx = (step['actual_voltage'] - click_x) / x_range
            dy = (step['mean_current'] - click_y) / y_range
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx
                
        self.select_step(closest_idx)

    def create_widgets(self):
        # ── TOP CONTROL PANEL (Database & Run Selection) ──
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Database label & entry
        ttk.Label(top_frame, text="SQLite DB Path:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.db_entry = ttk.Entry(top_frame, width=50)
        self.db_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Browse and load buttons
        ttk.Button(top_frame, text="Browse...", command=self.browse_db).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(top_frame, text="Load DB", command=lambda: self.load_database(self.db_entry.get())).grid(row=0, column=3, padx=5, pady=5)
        
        # Run Selection
        ttk.Label(top_frame, text="Experimental Run:", font=("Segoe UI", 10, "bold")).grid(row=0, column=4, sticky=tk.W, padx=15, pady=5)
        self.run_combo = ttk.Combobox(top_frame, width=50, state="readonly")
        self.run_combo.grid(row=0, column=5, padx=5, pady=5)
        self.run_combo.bind("<<ComboboxSelected>>", self.on_run_selected)
        
        # ── MAIN LAYOUT (Columns: Info Sidebar & Plots Main) ──
        main_layout = ttk.Frame(self)
        main_layout.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        # LEFT PANEL (Info & Stats)
        left_panel = ttk.Frame(main_layout, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Step Details card
        card_details = ttk.LabelFrame(left_panel, text="  Current Step Info  ", padding=15)
        card_details.pack(fill=tk.X, pady=(0, 15))
        
        self.lbl_sol = ttk.Label(card_details, text="Solution: -", font=("Segoe UI", 10, "bold"), wraplength=280)
        self.lbl_sol.pack(anchor=tk.W, pady=3)
        self.lbl_flow = ttk.Label(card_details, text="Flow Rate: -", font=("Segoe UI", 10))
        self.lbl_flow.pack(anchor=tk.W, pady=3)
        
        ttk.Separator(card_details, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        self.lbl_step_num = ttk.Label(card_details, text="Step - of -", font=("Segoe UI", 11, "bold"), foreground="#2e62d4")
        self.lbl_step_num.pack(anchor=tk.W, pady=3)
        self.lbl_target_v = ttk.Label(card_details, text="Target Voltage: -", font=("Segoe UI", 10))
        self.lbl_target_v.pack(anchor=tk.W, pady=3)
        self.lbl_actual_v = ttk.Label(card_details, text="Actual Voltage: -", font=("Segoe UI", 10))
        self.lbl_actual_v.pack(anchor=tk.W, pady=3)
        self.lbl_mean_i = ttk.Label(card_details, text="Average Current: -", font=("Segoe UI", 10, "bold"))
        self.lbl_mean_i.pack(anchor=tk.W, pady=3)
        self.lbl_ratio = ttk.Label(card_details, text="Ratio to Prev: -", font=("Segoe UI", 10, "bold"))
        self.lbl_ratio.pack(anchor=tk.W, pady=3)
        
        # Classifier card
        card_classifier = ttk.LabelFrame(left_panel, text="  Classifications  ", padding=15)
        card_classifier.pack(fill=tk.X, pady=(0, 15))
        
        self.lbl_img_class = ttk.Label(card_classifier, text="Image Classification:\n  -", font=("Segoe UI", 11, "bold"), wraplength=280)
        self.lbl_img_class.pack(anchor=tk.W, pady=5)
        self.lbl_man_class = ttk.Label(card_classifier, text="Manual Classification:\n  -", font=("Segoe UI", 10, "bold"), wraplength=280)
        self.lbl_man_class.pack(anchor=tk.W, pady=5)
        
        ttk.Separator(card_classifier, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        self.lbl_rf_class = ttk.Label(card_classifier, text="RF Spray Mode: -", font=("Segoe UI", 9))
        self.lbl_rf_class.pack(anchor=tk.W, pady=2)
        self.lbl_xgb_class = ttk.Label(card_classifier, text="XGB Spray Mode: -", font=("Segoe UI", 9))
        self.lbl_xgb_class.pack(anchor=tk.W, pady=2)
        
        # Hypothesis verification text box
        card_hypo = ttk.LabelFrame(left_panel, text="  Hypothesis Verification  ", padding=10)
        card_hypo.pack(fill=tk.BOTH, expand=True)
        
        self.hypo_text = tk.Text(
            card_hypo, bg="#1e1e1e", fg="#e0e0e0", bd=0, wrap=tk.WORD, 
            font=("Segoe UI", 9), padx=5, pady=5
        )
        self.hypo_text.pack(fill=tk.BOTH, expand=True)
        
        # Text tags for rich formatting
        self.hypo_text.tag_configure("title", font=("Segoe UI", 10, "bold"), foreground="#2e62d4")
        self.hypo_text.tag_configure("success", font=("Segoe UI", 9, "bold"), foreground="#2ecc71")
        self.hypo_text.tag_configure("warning", font=("Segoe UI", 9, "bold"), foreground="#f1c40f")
        self.hypo_text.tag_configure("danger", font=("Segoe UI", 9, "bold"), foreground="#e74c3c")
        
        # RIGHT PANEL (Plots)
        right_panel = ttk.Frame(main_layout)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Initialize Matplotlib Figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 7), dpi=100)
        self.fig.patch.set_facecolor('#121212')
        self.fig.subplots_adjust(hspace=0.45, top=0.92, bottom=0.1)
        
        # Initial subplots styling
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor('#1e1e1e')
            ax.spines['bottom'].set_color('#333333')
            ax.spines['top'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            ax.spines['right'].set_color('#333333')
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.1, color="#777777")
            
        self.ax1.set_title("Experimental I-V Curve (Average Current vs Voltage)", fontsize=11, color="white", fontweight="bold")
        self.ax2.set_title("Step Waveforms", fontsize=11, color="white", fontweight="bold")
        
        # Embed Figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot click listener
        self.canvas.mpl_connect('button_press_event', self.on_plot_clicked)
        
        # ── BOTTOM CONTROLS (Navigation buttons & shortcuts info) ──
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Navigation
        btn_prev = ttk.Button(bottom_frame, text="◀ Previous (P)", command=self.prev_step, width=15)
        btn_prev.pack(side=tk.LEFT, padx=5)
        
        btn_next = ttk.Button(bottom_frame, text="Next (N) ▶", command=self.next_step, width=15)
        btn_next.pack(side=tk.LEFT, padx=5)
        
        # Keyboard shortcut label helper
        lbl_shortcuts = ttk.Label(
            bottom_frame, 
            text="Shortcuts: [N] Next Step  |  [P] Previous Step  |  Click points on the I-V plot to select them directly", 
            font=("Segoe UI", 9), foreground="#888888"
        )
        lbl_shortcuts.pack(side=tk.RIGHT, padx=5)

if __name__ == "__main__":
    app = App()
    app.mainloop()
