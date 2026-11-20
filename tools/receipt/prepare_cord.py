#!/usr/bin/env python3
"""CORD 转换命令；未安装时只引导当前独立仓库 src。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from receipt.sources import cord_main

if __name__ == "__main__":
    cord_main()
