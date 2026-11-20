#!/usr/bin/env python3
"""事实清单和 split 隔离审计命令。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from receipt.data import audit_main

if __name__ == "__main__":
    audit_main()
