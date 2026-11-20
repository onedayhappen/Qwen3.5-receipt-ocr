#!/usr/bin/env python3
"""小规模 synthetic CPU 夹具生成命令，不构造业务训练数据。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from receipt.data import fixtures_main

if __name__ == "__main__":
    fixtures_main()
