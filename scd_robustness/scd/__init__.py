"""Robustness / calibration study of 3DGS scene change detection (CSE 598 Group 13).

Modules that run inside the method environments (O-SCD: Python 3.12, MV3DCD: Python 3.8)
use only the standard library, NumPy and OpenCV: runner, evaluate, metrics.
Planning and variant generation additionally need PyYAML (scd-tools environment).
"""

SCENES = [
    "Cantina", "Lounge", "Printing_area", "Lunch_room", "Meeting_room",   # indoor
    "Garden", "Pots", "Zen", "Playground", "Porch",                        # outdoor
]
INSTANCES = ["Instance_1", "Instance_2"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
