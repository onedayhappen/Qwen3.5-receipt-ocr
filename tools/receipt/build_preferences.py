#!/usr/bin/env python3
"""审定候选或显式合成负例的偏好构建命令。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from receipt.data import preferences_main

if __name__ == "__main__":
    preferences_main()
