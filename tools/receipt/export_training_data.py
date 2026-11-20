#!/usr/bin/env python3
"""SFT/GRPO/evaluation 导出命令。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from receipt.data import export_main

if __name__ == "__main__":
    export_main()
