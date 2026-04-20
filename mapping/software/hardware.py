"""
Hardware abstraction layer.

Owns all device handles (FUG power supply, syringe pump, oscilloscope)
and exposes a clean interface to the rest of the application.
"""

import sys
import time
import libtiepie
from mapping.software import configuration_tiepie
from mapping.software.FUG_functions  import (FUG_initialize, FUG_sendcommands,
                             get_voltage_from_PS, get_current_from_PS)
from mapping.software.PUMP_functions import (PUMP_initialize, set_pump_direction,
                             set_inner_diameter, set_flowrate,
                             start_pumping, stop_pumping,
                             low_motor_noize, beep_command)


class Hardware:

    SAMPLING_FREQ = 1e5   # Hz  (100 kHz → 50 k samples = 0.5 s)

    def __init__(self, cfg: dict):
        meas = cfg["typeofmeasurement"]
        self._slope = meas["slope"]
        self._v_start = meas["voltage_start"]
        self._init_fug(cfg["fug_com_port"])
        self._init_pump(cfg["pump_com_port"], cfg["diameter syringe"])
        self._init_scope()

    # ── FUG power supply ───────────────────────────────────────────────

    def _init_fug(self, port_idx: int):
        self.fug = FUG_initialize(port_idx)
        if self.fug is None:
            print("[HARDWARE] FATAL: Cannot connect to FUG power supply")
            sys.exit(1)
        FUG_sendcommands(self.fug, [
            ">S1B 0", "I 600e-6", ">S0B 0",
            f">S0R {self._slope}",
            f"U {self._v_start}", "F1"
        ])
        print("[HARDWARE] FUG ready")

    def set_voltage(self, volts: float):
        FUG_sendcommands(self.fug, [f"U {volts}"])

    def actual_voltage(self) -> float:
        return get_voltage_from_PS(self.fug)

    def actual_current(self) -> float:
        return get_current_from_PS(self.fug)

    # ── Syringe pump ───────────────────────────────────────────────────

    def _init_pump(self, port_idx: int, syringe_diameter: str):
        self.pump = PUMP_initialize(port_idx)
        if self.pump is None:
            print("[HARDWARE] FATAL: Cannot connect to pump")
            sys.exit(1)
        set_pump_direction(self.pump, "INF")
        set_inner_diameter(self.pump, syringe_diameter)
        low_motor_noize(self.pump)
        print("[HARDWARE] Pump ready")

    def set_flow_rate(self, flow_rate_ul_min: str):
        """Set flow rate (string, µL/min) and start pumping."""
        set_flowrate(self.pump, str(flow_rate_ul_min), "UM")
        time.sleep(0.5)
        start_pumping(self.pump)

    def stop_flow_rate(self):
        stop_pumping(self.pump)

    def pump_beep(self):
        beep_command(self.pump)

    # ── Oscilloscope ───────────────────────────────────────────────────

    def _init_scope(self):
        libtiepie.network.auto_detect_enabled = True
        libtiepie.device_list.update()
        self.scp = None
        for item in libtiepie.device_list:
            if item.can_open(libtiepie.DEVICETYPE_OSCILLOSCOPE):
                self.scp = item.open_oscilloscope()
                break
        if self.scp is None:
            print("[HARDWARE] FATAL: No oscilloscope found")
            self.set_voltage(0)
            sys.exit(1)
        self.scp = configuration_tiepie.config_TiePieScope(
            self.scp, self.SAMPLING_FREQ
        )
        print("[HARDWARE] Oscilloscope ready")

    # ── Shutdown ───────────────────────────────────────────────────────

    def shutdown(self):
        """Safely zero voltage and stop the pump before exit."""
        try:
            self.set_voltage(0)
        except Exception:
            pass
        try:
            stop_pumping(self.pump)
        except Exception:
            pass
        print("[HARDWARE] Hardware safely shut down")