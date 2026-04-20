"""
Camera control: Arduino shutter trigger + image classification.

Plug in your own model and capture logic in the two marked sections.
If no Arduino or model is available the module runs silently and
returns "N/A" so the rest of the program is unaffected.
"""

import time
import serial
import serial.tools.list_ports


class CameraClassifier:

    def __init__(self, com_port_idx: int, model_path: str | None = None):
        self.arduino = self._connect_arduino(com_port_idx)
        self.model   = self._load_model(model_path)

    # ── Arduino trigger ───────────────────────────────────────────────

    @staticmethod
    def _connect_arduino(com_port_idx: int):
        ports = list(serial.tools.list_ports.comports())
        if not ports or com_port_idx >= len(ports):
            print("[CAMERA] No Arduino found – running without camera")
            return None
        try:
            ard = serial.Serial(
                port=ports[com_port_idx].device,
                baudrate=9600,
                timeout=1
            )
            time.sleep(2)   # wait for Arduino reset
            print(f"[CAMERA] Arduino connected: {ports[com_port_idx].device}")
            return ard
        except Exception as e:
            print(f"[CAMERA] Could not connect to Arduino: {e}")
            return None

    def _trigger(self):
        if self.arduino is None:
            return
        try:
            self.arduino.write(b"1\n")
            self.arduino.flush()
            print(f"[CAMERA] Trigger sent")
        except Exception as e:
            print(f"[CAMERA] Trigger error: {e}")

    # ── Image model ───────────────────────────────────────────────────

    @staticmethod
    def _load_model(model_path: str | None):
        """
        ── PLUG IN YOUR MODEL LOADER HERE ──────────────────────────────
        Expected interface:  model.predict(image_array) -> str label

        Example (joblib):
            from joblib import load
            return load(model_path)

        Example (ONNX):
            import onnxruntime as ort
            return ort.InferenceSession(model_path)
        ────────────────────────────────────────────────────────────────
        """
        if model_path is None:
            return None
        try:
            # from joblib import load
            # return load(model_path)
            print(f"[CAMERA] Model path set ({model_path}) – add loader in camera.py")
            return None
        except Exception as e:
            print(f"[CAMERA] Could not load image model: {e}")
            return None

    # ── Frame capture ─────────────────────────────────────────────────

    @staticmethod
    def _capture_frame():
        """
        ── PLUG IN YOUR CAPTURE CODE HERE ──────────────────────────────
        Return a numpy array (H, W, C), or None on failure.

        Example (OpenCV USB camera):
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        ────────────────────────────────────────────────────────────────
        """
        return None

    def _classify_frame(self, frame) -> str:
        if self.model is None or frame is None:
            return "N/A"
        try:
            return str(self.model.predict(frame))
        except Exception as e:
            print(f"[CAMERA] Classification error: {e}")
            return "Error"

    # ── Public API ────────────────────────────────────────────────────

    def capture_and_classify(self) -> str:
        """Trigger shutter, grab a frame, classify. Returns a label string."""
        self._trigger()
        print("[CAMERA] Trigger sent")
        frame = self._capture_frame()
        return self._classify_frame(frame)