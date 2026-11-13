"""
PostureGuard Standalone Executable Builder (PyInstaller)
-------------------------------------------------------
Packages PostureGuard into a standalone Windows desktop executable
with bundled model weights, MediaPipe dependencies, and system tray support.
"""

from pathlib import Path
import sys
import PyInstaller.__main__

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "posture_model.joblib"

if not MODEL_FILE.exists():
    print(f"Error: Model bundle not found at {MODEL_FILE}", file=sys.stderr)
    sys.exit(1)

print("=" * 60)
print("Building PostureGuard Standalone Executable...")
print("=" * 60)

args = [
    str(PROJECT_ROOT / "run_desktop.py"),
    "--name=PostureGuard",
    "--windowed",
    "--noconsole",
    f"--add-data={MODEL_FILE};models",
    "--collect-all=mediapipe",
    "--collect-all=pystray",
    "--collect-all=PIL",
    "--clean",
    "--noconfirm",
]

PyInstaller.__main__.run(args)
print("\nBuild complete! Executable is located in dist/PostureGuard/PostureGuard.exe")
