"""Assemble the public Kaggle notebook from the repo's source files.

The notebook writes the src modules into /kaggle/working/src, symlinks the
competition input to data/raw, runs extraction -> Combine metrics -> figures,
and displays the figures. Everything the writeup shows is produced here.

Usage: python kaggle/build_final_notebook.py  (then `kaggle kernels push -p kaggle/final`)
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "kaggle" / "final"
FINAL.mkdir(exist_ok=True)

SRC_FILES = ["trench_features.py", "build_combine_trench_metrics.py", "trench_analysis.py", "make_figures.py"]
EXTRACT_BODY = (ROOT / "kaggle" / "extract" / "extract_body.py").read_text(encoding="utf-8")

INTRO = """# Same Yardstick: measuring trench movement identically at the Combine and after the snap

**NFL Big Data Bowl 2027 — companion notebook for the writeup.** Everything in the writeup (numbers, models, figures) is produced by this notebook from the competition files only.

Pipeline:
1. One feature function (`trench_features.py`) measures depth, lateral movement, speed and timing in the first 1.5 s of movement — applied to Combine drill attempts (`PASS_RUSH_DROP` for OL, `PASS_RUSH_DRILL` for DL, plus the 40) and to every regular-season pass-pro / pass-rush snap of the 239 OL/DL prospects, starting at `ball_snap`.
2. Reliability first (attempt 1 vs 2 at the Combine; odd vs even weeks in games).
3. Transfer: same metric, Combine → snaps, with height, weight and draft slot held fixed (bootstrap CIs).
4. Outcome: in-game mechanic → pressure (DL) / pressure allowed (OL), with situational controls and within-player (fixed-effect) models.

Runs in ~3 minutes on CPU."""

SETUP = '''import glob, os, sys, shutil
from pathlib import Path
BASE = os.path.dirname(glob.glob("/kaggle/input/**/games.csv", recursive=True)[0]) + "/"
ROOT = Path("/kaggle/working"); (ROOT / "src").mkdir(exist_ok=True); (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
if not os.path.lexists(ROOT / "data" / "raw"):
    os.symlink(BASE.rstrip("/"), ROOT / "data" / "raw")
sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
print("BASE", BASE)'''


def write_cell(name: str, text: str) -> str:
    assert "'''" not in text, name
    return f"Path('src/{name}').write_text(r'''{text}''', encoding='utf-8')\nprint('wrote src/{name}')"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETUP)]
    cells.append(nbf.v4.new_markdown_cell("## Source modules (also at the GitHub repo linked in the writeup)"))
    for f in SRC_FILES:
        cells.append(nbf.v4.new_code_cell(write_cell(f, (ROOT / "src" / f).read_text(encoding="utf-8"))))
    cells.append(nbf.v4.new_markdown_cell("## 1. In-game extraction: first 3 s after `ball_snap` for every OL/DL prospect, regular season only"))
    extract_cell = "from trench_features import *\n" + EXTRACT_BODY + "\nfor f in ['trench_game_features.csv', 'trench_game_profiles.parquet', 'extract_diagnostics.csv']:\n    shutil.move(f, 'data/processed/' + f)"
    cells.append(nbf.v4.new_code_cell(extract_cell))
    cells.append(nbf.v4.new_markdown_cell("## 2. Combine drill metrics with the same function"))
    cells.append(nbf.v4.new_code_cell("import build_combine_trench_metrics\nbuild_combine_trench_metrics.main()"))
    cells.append(nbf.v4.new_markdown_cell("## 3. Reliability, transfer, outcome models and figures"))
    cells.append(nbf.v4.new_code_cell("import make_figures\nmake_figures.main()"))
    cells.append(nbf.v4.new_code_cell("from IPython.display import Image, display\nfor i in range(1, 8):\n    display(Image(filename=sorted(glob.glob(f'output/figures/fig{i}_*.png'))[0]))"))
    cells.append(nbf.v4.new_markdown_cell("## 4. All numbers used in the writeup"))
    cells.append(nbf.v4.new_code_cell("import json\nprint(json.dumps(json.load(open('output/results.json')), indent=1))"))
    nb["cells"] = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, FINAL / "bdb27-same-yardstick.ipynb")
    (FINAL / "kernel-metadata.json").write_text(
        '{\n  "id": "vanductan/bdb27-same-yardstick-combine-vs-snap",\n  "title": "BDB27 same yardstick combine vs snap",\n'
        '  "code_file": "bdb27-same-yardstick.ipynb",\n  "language": "python",\n  "kernel_type": "notebook",\n  "is_private": false,\n'
        '  "enable_gpu": false,\n  "enable_tpu": false,\n  "enable_internet": false,\n  "dataset_sources": [],\n'
        '  "competition_sources": ["nfl-big-data-bowl-2027"],\n  "kernel_sources": [],\n  "model_sources": []\n}\n', encoding="utf-8")
    print("wrote", FINAL / "bdb27-same-yardstick.ipynb", len(cells), "cells")


if __name__ == "__main__":
    main()
