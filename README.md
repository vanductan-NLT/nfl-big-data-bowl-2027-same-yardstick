# Same Yardstick — NFL Big Data Bowl 2027

Measuring trench (OL/DL) movement identically at the NFL Scouting Combine and in the first 1.5 s after `ball_snap`, then testing three links separately: does a metric repeat, does it carry from the Combine to NFL snaps, and does the in-game version predict pressure.

- Competition: https://www.kaggle.com/competitions/nfl-big-data-bowl-2027 (Open Track)
- Submitted writeup: https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/writeups/same-yardstick-measuring-trench-movement-identica
- Public notebook (reproduces every number and figure): https://www.kaggle.com/code/vanductan/bdb27-same-yardstick-combine-vs-snap
- Writeup text: `docs/writeup-same-yardstick.md`; figures: `output/figures/`; numbers: `output/results.json`

## Layout

```
src/trench_features.py              one polars feature function used on both Combine and game frames
src/build_combine_trench_metrics.py Combine drill attempts -> data/processed/combine_trench_attempts.csv (+ profiles)
src/build_combine_player_side_table.py per-player, per-side Combine table (exploration)
src/trench_analysis.py              player means, split-half reliability, bootstrap/partial correlations
src/make_figures.py                 all writeup numbers (results.json) and figures 1-7
src/explore_*.py                    exploration scripts (side-matched transfer, DL models)
kaggle/extract/                     heavy in-game extraction script run on Kaggle (2.1 GB tracking -> 42 MB features)
kaggle/build_kaggle_script.py       concatenates the feature module + extraction body into one Kaggle script
kaggle/build_final_notebook.py      assembles the public notebook from src/
plans/                              plan and reports
```

## Reproduce

Competition data is CC BY-NC 4.0 and is **not** in this repo. Either run the public Kaggle notebook, or locally:

```bash
pip install polars pandas numpy scipy statsmodels matplotlib pyarrow nbformat kaggle
kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw --unzip   # or only the small files + combine_tracking.csv
python kaggle/build_kaggle_script.py && kaggle kernels push -p kaggle/extract  # in-game extraction on Kaggle (~1 min)
kaggle kernels output vanductan/bdb27-trench-extract -p data/processed
python src/build_combine_trench_metrics.py
python src/make_figures.py
```

## Key numbers (65 OL / 69 DL with >=100 snaps)

- In-game set / get-off metrics: split-half reliability 0.92–0.98.
- OL kick-slide drill -> in-game set: depth at 1.0 s r = 0.21 (−0.01, 0.43) pre-specified; speed at 1 yd r = 0.40 (0.09, 0.62). 40-yard start and official times: ~0.
- DL pass-rush drill -> in-game get-off: ~0 (CI ±0.25). Draft slot -> pressure rate −0.33.
- DL in-game "second yard" (time 1 yd -> 2 yd) -> pressure rate −0.51 (−0.65, −0.34), beyond weight and draft slot; NGS get-off −0.44.
- OL set depth: no between-player relation to pressure allowed after controls; within a player, deeper-than-usual sets are the pressured ones.

License: code Apache-2.0.
