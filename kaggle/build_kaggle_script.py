"""Concatenate src/trench_features.py + kaggle/extract/extract_body.py into the
single-file Kaggle script (Kaggle scripts cannot import local modules).

Usage: python kaggle/build_kaggle_script.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
feat = (ROOT / "src" / "trench_features.py").read_text(encoding="utf-8")
body = (ROOT / "kaggle" / "extract" / "extract_body.py").read_text(encoding="utf-8")
target = ROOT / "kaggle" / "extract" / "bdb27_trench_extract.py"
target.write_text(feat + "\n" + body, encoding="utf-8")
print("wrote", target, len(feat) + len(body), "chars")
