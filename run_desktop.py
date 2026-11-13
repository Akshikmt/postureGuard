"""
PostureGuard Desktop Application Launcher
-----------------------------------------
Launches the background sitting posture monitor with system tray integration.
"""

from pathlib import Path
import sys

# Ensure postureGuard directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.desktop_app import main

if __name__ == "__main__":
    main()
