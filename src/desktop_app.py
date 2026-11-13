"""
PostureGuard: Silent Background Desktop Application
---------------------------------------------------
A discreet, system-tray sitting posture guardian that runs silently in the
background with zero camera window on the screen.

Features:
1. Headless Background Monitoring: Captures webcam frames and runs MediaPipe
   + 4-class posture inference strictly in memory with zero on-screen camera feed.
2. Smart Slouch Alert:
   - Triggers a floating warning popup ("⚠️ Sit Straight!") if slouching persists for >= 10s.
   - Automatically dismisses the popup once the user maintains upright posture for >= 5s.
3. Windows Startup Integration:
   - Registers/unregisters in HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run.
   - Automatically launches and resumes monitoring at system startup.
4. System Tray Integration (pystray):
   - Minimizes to system tray with quick-action context menu.
   - Displays posture status without disrupting user workflows.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, List, Optional, Tuple
import winreg

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import pystray
import sklearn
import sklearn.ensemble
import sklearn.tree
import sklearn.utils

# Project Root Resolution
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    PROJECT_ROOT = Path(sys._MEIPASS)
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import extract_posture_features
from src.landmarks import extract_key_landmarks
from src.realtime_posture import (
    PosturePredictor,
    PostureStateHysteresis,
    UprightPostureCalibrator,
    DEFAULT_MODEL_PATH,
)

# Registry Key for Windows Startup
STARTUP_REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_NAME = "PostureGuard"
CONFIG_FILE = PROJECT_ROOT / "models" / "app_config.json"


class StartupManager:
    """Manages Windows Startup registration via CurrentUser Registry."""

    @staticmethod
    def is_enabled() -> bool:
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, STARTUP_REG_KEY, 0, winreg.KEY_READ)
            try:
                val, _ = winreg.QueryValueEx(key, APP_NAME)
                return bool(val)
            except FileNotFoundError:
                return False
            finally:
                winreg.CloseKey(key)
        except Exception:
            return False

    @staticmethod
    def set_enabled(enable: bool) -> bool:
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, STARTUP_REG_KEY, 0, winreg.KEY_ALL_ACCESS)
            if enable:
                if getattr(sys, "frozen", False):
                    # Compiled PyInstaller executable
                    cmd = f'"{sys.executable}" --background'
                else:
                    # Running from Python source
                    python_exe = sys.executable
                    # Prefer pythonw.exe if available to avoid opening a console window on boot
                    pythonw_exe = Path(python_exe).parent / "pythonw.exe"
                    runner = str(pythonw_exe) if pythonw_exe.exists() else python_exe
                    script_path = PROJECT_ROOT / "src" / "desktop_app.py"
                    cmd = f'"{runner}" "{script_path}" --background'
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
            return True
        except Exception as e:
            print(f"[StartupManager] Registry error: {e}", file=sys.stderr)
            return False


class PostureMonitorWorker(threading.Thread):
    """
    Background worker thread: captures webcam frames, extracts features,
    and runs posture classification strictly in memory (no GUI/camera window).
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        model_path: Optional[Path] = None,
        slouch_trigger_sec: float = 10.0,
        upright_dismiss_sec: float = 5.0,
    ):
        super().__init__(daemon=True, name="PostureMonitorWorker")
        self.event_queue = event_queue
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.slouch_trigger_sec = slouch_trigger_sec
        self.upright_dismiss_sec = upright_dismiss_sec

        self.running = True
        self.enabled = True

        # ML and Biomechanical Predictors
        self.predictor = PosturePredictor(self.model_path)
        self.hysteresis = PostureStateHysteresis(persistence_frames=5, min_confidence=0.55, min_margin=0.10)
        self.calibrator = UprightPostureCalibrator(target_duration_sec=3.0)

        # Posture Timing State Machine
        self.current_state: Optional[str] = None
        self.alert_active: bool = False
        self.slouch_start_time: Optional[float] = None
        self.upright_start_time: Optional[float] = None

    def stop(self):
        """Signals the background worker to stop and release camera."""
        self.running = False

    def trigger_calibration(self):
        """Starts 3-second upright baseline calibration sequence."""
        self.calibrator.start_calibration()

    def reset_calibration(self):
        """Resets calibration to uncalibrated default."""
        self.calibrator.cancel_or_reset()

    @staticmethod
    def _open_camera() -> Optional[cv2.VideoCapture]:
        """Tries multiple backends and device indices to reliably open the camera."""
        configs = [
            (0, cv2.CAP_DSHOW),
            (0, None),
            (1, cv2.CAP_DSHOW),
            (1, None),
            (2, None),
        ]
        for idx, backend in configs:
            try:
                cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        return cap
                    cap.release()
            except Exception:
                pass
        return None

    def run(self):
        cap = self._open_camera()
        if cap is None:
            self.event_queue.put({"type": "error", "message": "Could not access webcam device. Please check permissions."})
            return

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as pose:
            while self.running:
                if not self.enabled:
                    time.sleep(0.5)
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                # Headless MediaPipe Pose processing (zero cv2.imshow)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose.process(rgb_frame)

                now = time.time()
                detected = results.pose_landmarks is not None
                raw_pred = None
                probs: Dict[str, float] = {}

                if detected:
                    landmarks = extract_key_landmarks(results.pose_landmarks, min_visibility=0.4)
                    features = extract_posture_features(landmarks) if landmarks else None

                    # Calibration tracking
                    calib_info = self.calibrator.update(features, landmarks)
                    calibrated_d0 = (
                        self.calibrator.baseline.shoulder_distance
                        if (self.calibrator.state == UprightPostureCalibrator.STATE_CALIBRATED and self.calibrator.baseline)
                        else None
                    )
                    calibrated_ratio0 = (
                        self.calibrator.baseline.head_to_shoulder_ratio
                        if (self.calibrator.state == UprightPostureCalibrator.STATE_CALIBRATED and self.calibrator.baseline)
                        else None
                    )

                    # Model prediction + Hysteresis
                    raw_pred, probs, _ = self.predictor.predict(
                        features,
                        baseline_distance=calibrated_d0,
                        baseline_head_ratio=calibrated_ratio0
                    )
                    stabilized_label = self.hysteresis.update(raw_pred, probs)
                else:
                    self.hysteresis.clear()
                    stabilized_label = None
                    calib_info = self.calibrator.update(None, None)

                self.current_state = stabilized_label

                # Posture Alert State Machine
                self._handle_alert_timing(stabilized_label, now)

                # Send periodic telemetry to UI queue (10 FPS loop)
                self.event_queue.put({
                    "type": "telemetry",
                    "posture": stabilized_label or "No Person Detected",
                    "confidence": probs.get(stabilized_label, 0.0) if stabilized_label else 0.0,
                    "calib_state": self.calibrator.state,
                    "alert_active": self.alert_active,
                })

                time.sleep(0.08)  # ~12 FPS: Ultra-low CPU usage (<3%)

        cap.release()

    def _handle_alert_timing(self, label: Optional[str], now: float):
        """
        Implements 10s slouching trigger and 5s upright dismissal logic.
        """
        if not self.alert_active:
            # NORMAL MODE: Watch for continuous slouching >= 10s
            if label == "slouching":
                if self.slouch_start_time is None:
                    self.slouch_start_time = now
                slouch_sec = now - self.slouch_start_time
                if slouch_sec >= self.slouch_trigger_sec:
                    self.alert_active = True
                    self.upright_start_time = None
                    self.event_queue.put({"type": "show_alert"})
            else:
                self.slouch_start_time = None
        else:
            # ALERT ACTIVE MODE: Watch for continuous upright posture >= 5s
            if label == "correct":
                if self.upright_start_time is None:
                    self.upright_start_time = now
                upright_sec = now - self.upright_start_time
                self.event_queue.put({
                    "type": "update_alert",
                    "upright_sec": upright_sec,
                    "target_sec": self.upright_dismiss_sec
                })
                if upright_sec >= self.upright_dismiss_sec:
                    self.alert_active = False
                    self.slouch_start_time = None
                    self.upright_start_time = None
                    self.event_queue.put({"type": "hide_alert"})
            else:
                # If they slouch or lean during recovery, reset upright timer
                self.upright_start_time = None
                self.event_queue.put({
                    "type": "update_alert",
                    "upright_sec": 0.0,
                    "target_sec": self.upright_dismiss_sec
                })


class WarningPopup(tk.Toplevel):
    """
    Modern borderless floating toast alert that appears when slouching exceeds 10s
    and automatically vanishes when user maintains upright posture for 5s.
    """

    def __init__(self, master):
        super().__init__(master)
        self.withdraw()  # Start hidden
        self.overrideredirect(True)
        self.attributes("-topmost", True)

        # Style colors (Dark theme with crimson alert)
        self.bg_color = "#181825"
        self.border_color = "#F38BA8"
        self.text_color = "#CDD6F4"
        self.accent_color = "#FAB387"

        self.configure(bg=self.border_color)

        container = tk.Frame(self, bg=self.bg_color, padx=16, pady=14)
        container.pack(padx=2, pady=2, fill="both", expand=True)

        header_frame = tk.Frame(container, bg=self.bg_color)
        header_frame.pack(fill="x", expand=True)

        icon_label = tk.Label(header_frame, text="⚠️", font=("Segoe UI", 20), bg=self.bg_color, fg=self.border_color)
        icon_label.pack(side="left", padx=(0, 10))

        title_frame = tk.Frame(header_frame, bg=self.bg_color)
        title_frame.pack(side="left", fill="x", expand=True)

        title_label = tk.Label(
            title_frame,
            text="Sit Straight!",
            font=("Segoe UI", 13, "bold"),
            bg=self.bg_color,
            fg=self.border_color
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            title_frame,
            text="Slouching detected for over 10 seconds.",
            font=("Segoe UI", 9),
            bg=self.bg_color,
            fg="#BAC2DE"
        )
        subtitle_label.pack(anchor="w")

        # Dynamic Recovery Progress Bar & Label
        self.status_label = tk.Label(
            container,
            text="Sit upright for 5s to dismiss (0.0s / 5.0s)",
            font=("Segoe UI", 9, "italic"),
            bg=self.bg_color,
            fg=self.accent_color,
            pady=6
        )
        self.status_label.pack(anchor="w")

        self.progress_bar = ttk.Progressbar(container, orient="horizontal", mode="determinate", maximum=5.0)
        self.progress_bar.pack(fill="x", pady=(2, 4))

        # Position at top-right of screen
        self.width = 360
        self.height = 125
        screen_w = self.winfo_screenwidth()
        x_pos = screen_w - self.width - 24
        y_pos = 40
        self.geometry(f"{self.width}x{self.height}+{x_pos}+{y_pos}")

    def show_alert(self):
        """Displays the alert window on screen."""
        self.progress_bar["value"] = 0.0
        self.status_label.config(text="Sit upright for 5s to dismiss (0.0s / 5.0s)")
        self.deiconify()
        self.lift()

    def hide_alert(self):
        """Hides the alert window."""
        self.withdraw()

    def update_recovery(self, upright_sec: float, target_sec: float = 5.0):
        """Updates recovery countdown and progress bar."""
        val = min(target_sec, max(0.0, upright_sec))
        self.progress_bar["value"] = val
        pct = int((val / target_sec) * 100)
        self.status_label.config(
            text=f"Holding upright: {val:.1f}s / {target_sec:.1f}s ({pct}%)"
        )


class PostureGuardApp:
    """Main Desktop Control Window and System Tray Coordinator."""

    def __init__(self, root: tk.Tk, start_in_background: bool = False):
        self.root = root
        self.root.title("PostureGuard Desktop")
        self.root.geometry("440x520")
        self.root.resizable(False, False)
        self.root.configure(bg="#1E1E2E")

        # Set window icon
        icon_path = PROJECT_ROOT / "posture_icon.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except Exception:
                pass

        self.event_queue = queue.Queue()
        self.worker = PostureMonitorWorker(self.event_queue)
        self.popup = WarningPopup(self.root)

        self._build_ui()
        self._setup_tray()

        # Handle window minimize / close
        self.root.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)

        # Start worker thread
        self.worker.start()

        # Start UI event poll
        self.root.after(50, self._process_events)

        if start_in_background:
            self.root.withdraw()

    def _build_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#181825", padx=20, pady=16)
        header.pack(fill="x")

        title = tk.Label(header, text="🛡️ PostureGuard AI", font=("Segoe UI", 16, "bold"), fg="#CDD6F4", bg="#181825")
        title.pack(anchor="w")
        subtitle = tk.Label(header, text="Silent background sitting posture monitor", font=("Segoe UI", 9), fg="#A6ADC8", bg="#181825")
        subtitle.pack(anchor="w")

        # Main Content Frame
        content = tk.Frame(self.root, bg="#1E1E2E", padx=20, pady=16)
        content.pack(fill="both", expand=True)

        # Status Card
        status_card = tk.LabelFrame(content, text=" System Status ", font=("Segoe UI", 10, "bold"), fg="#89B4FA", bg="#1E1E2E", padx=12, pady=10)
        status_card.pack(fill="x", pady=(0, 14))

        self.lbl_monitoring = tk.Label(status_card, text="🟢 Monitoring Active (Background)", font=("Segoe UI", 10, "bold"), fg="#A6E3A1", bg="#1E1E2E")
        self.lbl_monitoring.pack(anchor="w", pady=2)

        self.lbl_posture = tk.Label(status_card, text="Current Posture: Detecting...", font=("Segoe UI", 10), fg="#CDD6F4", bg="#1E1E2E")
        self.lbl_posture.pack(anchor="w", pady=2)

        self.lbl_alert_rule = tk.Label(status_card, text="Rule: Alert after 10s slouching | Dismiss after 5s upright", font=("Segoe UI", 8), fg="#9399B2", bg="#1E1E2E")
        self.lbl_alert_rule.pack(anchor="w", pady=(4, 0))

        # Controls Section
        btn_frame = tk.Frame(content, bg="#1E1E2E")
        btn_frame.pack(fill="x", pady=6)

        self.btn_toggle = tk.Button(
            btn_frame,
            text="⏸️ Disable Monitoring",
            font=("Segoe UI", 10, "bold"),
            bg="#F38BA8",
            fg="#11111B",
            activebackground="#EBA0AC",
            relief="flat",
            padx=14,
            pady=8,
            command=self.toggle_monitoring
        )
        self.btn_toggle.pack(fill="x", pady=4)

        self.btn_calibrate = tk.Button(
            btn_frame,
            text="🎯 Calibrate Upright Baseline (3s)",
            font=("Segoe UI", 10),
            bg="#89B4FA",
            fg="#11111B",
            activebackground="#B4BEFE",
            relief="flat",
            padx=14,
            pady=8,
            command=self.trigger_calibration
        )
        self.btn_calibrate.pack(fill="x", pady=4)

        # Settings Section
        settings_card = tk.LabelFrame(content, text=" Preferences ", font=("Segoe UI", 10, "bold"), fg="#89B4FA", bg="#1E1E2E", padx=12, pady=10)
        settings_card.pack(fill="x", pady=(10, 0))

        self.var_startup = tk.BooleanVar(value=StartupManager.is_enabled())
        self.chk_startup = tk.Checkbutton(
            settings_card,
            text="Launch automatically on Windows startup",
            variable=self.var_startup,
            command=self.on_toggle_startup,
            font=("Segoe UI", 9),
            fg="#CDD6F4",
            bg="#1E1E2E",
            activebackground="#1E1E2E",
            activeforeground="#CDD6F4",
            selectcolor="#181825"
        )
        self.chk_startup.pack(anchor="w", pady=2)

        # Bottom Action Bar
        bottom_frame = tk.Frame(self.root, bg="#181825", padx=20, pady=10)
        bottom_frame.pack(fill="x", side="bottom")

        btn_hide = tk.Button(
            bottom_frame,
            text="Hide to System Tray",
            font=("Segoe UI", 9),
            bg="#313244",
            fg="#CDD6F4",
            activebackground="#45475A",
            relief="flat",
            command=self.minimize_to_tray
        )
        btn_hide.pack(side="left")

        btn_exit = tk.Button(
            bottom_frame,
            text="Exit App",
            font=("Segoe UI", 9),
            bg="#313244",
            fg="#F38BA8",
            activebackground="#45475A",
            relief="flat",
            command=self.exit_app
        )
        btn_exit.pack(side="right")

    def _create_tray_icon(self, active: bool = True) -> Image.Image:
        """Generates a sleek 64x64 RGBA system tray icon."""
        img = Image.new("RGBA", (64, 64), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        fill_col = (166, 227, 161, 255) if active else (147, 153, 178, 255)
        # Draw posture shield silhouette
        draw.ellipse((10, 10, 54, 54), fill=fill_col)
        draw.ellipse((22, 16, 42, 36), fill=(24, 24, 37, 255))
        draw.ellipse((16, 38, 48, 56), fill=(24, 24, 37, 255))
        return img

    def _setup_tray(self):
        """Initializes pystray system tray icon in a dedicated daemon thread."""
        icon_img = self._create_tray_icon(active=True)
        menu = pystray.Menu(
            pystray.MenuItem("Open PostureGuard", self.show_window, default=True),
            pystray.MenuItem("Toggle Monitoring", self.toggle_monitoring),
            pystray.MenuItem("Calibrate Upright Baseline", self.trigger_calibration),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self.exit_app)
        )
        self.tray_icon = pystray.Icon("PostureGuard", icon_img, "PostureGuard AI", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def _process_events(self):
        """Reads worker telemetry and alert commands on the Tkinter main thread."""
        try:
            while not self.event_queue.empty():
                evt = self.event_queue.get_nowait()
                msg_type = evt.get("type")

                if msg_type == "telemetry":
                    posture = evt.get("posture", "Unknown")
                    conf = evt.get("confidence", 0.0) * 100
                    color_map = {
                        "correct": "#A6E3A1",
                        "slouching": "#F38BA8",
                        "leaning_left": "#89DCEB",
                        "leaning_right": "#CBA6F7",
                        "No Person Detected": "#6C7086"
                    }
                    col = color_map.get(posture, "#CDD6F4")
                    self.lbl_posture.config(
                        text=f"Current Posture: {posture.replace('_', ' ').title()} ({conf:.0f}%)",
                        fg=col
                    )
                elif msg_type == "show_alert":
                    self.popup.show_alert()
                elif msg_type == "hide_alert":
                    self.popup.hide_alert()
                elif msg_type == "update_alert":
                    self.popup.update_recovery(evt.get("upright_sec", 0.0), evt.get("target_sec", 5.0))
                elif msg_type == "error":
                    messagebox.showerror("PostureGuard Error", evt.get("message", "Unknown error"))
        except Exception:
            pass

        self.root.after(50, self._process_events)

    def toggle_monitoring(self, icon=None, item=None):
        """Toggles background monitoring on/off."""
        self.worker.enabled = not self.worker.enabled
        if self.worker.enabled:
            self.lbl_monitoring.config(text="🟢 Monitoring Active (Background)", fg="#A6E3A1")
            self.btn_toggle.config(text="⏸️ Disable Monitoring", bg="#F38BA8")
            self.tray_icon.icon = self._create_tray_icon(active=True)
            # Ensure startup is active when enabled
            if self.var_startup.get():
                StartupManager.set_enabled(True)
        else:
            self.lbl_monitoring.config(text="⏸️ Monitoring Paused (Disabled)", fg="#F9E2AF")
            self.btn_toggle.config(text="▶️ Enable Monitoring", bg="#A6E3A1")
            self.lbl_posture.config(text="Current Posture: Paused", fg="#6C7086")
            self.popup.hide_alert()
            self.tray_icon.icon = self._create_tray_icon(active=False)

    def trigger_calibration(self, icon=None, item=None):
        """Triggers upright posture baseline calibration."""
        if not self.worker.enabled:
            messagebox.showinfo("PostureGuard", "Please enable monitoring before calibrating.")
            return
        self.worker.trigger_calibration()
        messagebox.showinfo(
            "Calibration Started",
            "Sit upright in your natural ergonomic position for 3 seconds.\n"
            "The system is capturing your baseline..."
        )

    def on_toggle_startup(self):
        """Handles user clicking the Startup checkbox."""
        enabled = self.var_startup.get()
        success = StartupManager.set_enabled(enabled)
        if not success:
            messagebox.showwarning("Startup Error", "Could not modify Windows startup registry.")

    def minimize_to_tray(self):
        """Hides main window while keeping background monitoring active."""
        self.root.withdraw()

    def show_window(self, icon=None, item=None):
        """Restores main window from tray."""
        self.root.deiconify()
        self.root.lift()

    def exit_app(self, icon=None, item=None):
        """Completely terminates background thread, tray icon, and app."""
        self.worker.stop()
        if hasattr(self, "tray_icon") and self.tray_icon:
            self.tray_icon.stop()
        self.popup.destroy()
        self.root.destroy()
        sys.exit(0)


def main():
    start_background = "--background" in sys.argv
    root = tk.Tk()
    app = PostureGuardApp(root, start_in_background=start_background)
    root.mainloop()


if __name__ == "__main__":
    main()
