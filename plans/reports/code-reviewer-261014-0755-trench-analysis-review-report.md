# Trench analysis code review (BDB 2027, read-only)

Scope: `src/trench_features.py`, `kaggle/extract/extract_body.py`, `src/build_combine_trench_metrics.py`, `src/trench_analysis.py`, `src/make_figures.py`, `kaggle/build_final_notebook.py`. Verified against `data/raw/combine_tracking.csv`, `data/processed/trench_game_features.csv`, `output/results.json`, `docs/writeup-same-yardstick.md`. Local env: pandas 3.0.1, polars 2.0.0.

## Verified correct (empirical)

- Depth sign: in-game OL median depth_10 = +0.95, DL = +1.62, identical for play_direction left/right. Combine PASS_RUSH_DROP net dx < 0 in 100% of attempts (ux=-1 correct); PASS_RUSH_DRILL early-burst dir median 84 deg (+x), so ux=+1 is right for the first second even though the full drill arcs ~17 yd in y.
- Lateral sign: on 31k OL / 19k DL snaps, P(lat<0 | moving +y) = 1.0 for play right and 0.0 for play left. So lat<0 = OL's own left, and for DL lat<0 = offensive left = DL's own right. Same for Combine DL (faces +x, ux=+1). The comment at `trench_features.py:81` is correct only read as "offensive left"; `side_10` for DL is not the rusher's own side.
- Orientation: `face` (90 for +x, 270 for -x) matches the NFL convention (0 = +y, clockwise); orel_00 median 10 deg (OL) / 21 deg (DL). A wrong convention would give ~180.
- `play_nullified_by_penalty` is 'Y'/'N' (filter works); `quarter`=5 removed; bool columns arrive as Python bool/NaN objects and `map`/`fillna(0.0)`+`to_numeric` resolve to 0/1 correctly on pandas 3.0 (tested). No duplicate (game,play,nfl_id) keys; no duplicate nfl_id in players/combine_results; `lined_up_position` uses T/G/C (fig6 filter works).
- Spearman-Brown `2r/(1+r)` correct; partial correlation via residualize-both-then-Pearson is correct; `games_for_r70` formula correct.

## Critical

### C1. Combine "movement onset" is truncated by recording start; reported Combine numbers are affected
`build_combine_trench_metrics.py:26,70` defines t=0 as the first frame with s >= 0.5. Measured on raw tracking, the first recorded frame already has s >= 0.5 in:

| drill | 2023 | 2024 | 2025 |
|---|---|---|---|
| FORTY_YARD_DASH | 87% | 91% | 81% |
| PASS_RUSH_DRILL (DL) | 58% | 78% | 4% |
| PASS_RUSH_DROP (OL) | 10% | 11% | 7% (but first-frame s median 0.2-0.4, p90 ~0.5: already moving) |

Failure scenario: for the 40 and the 2023/24 DL drill, t=0 is "whenever the recorder started" (first-frame s median 0.62, p90 up to 1.04 yd/s, i.e. 0.1-0.2 s into the start). `t_d1` at the Combine is ~0.6 s, so a 0.1-0.2 s offset is 20-35% error per attempt, uncorrelated between attempts. This mechanically deflates: (a) Fig 2 "Combine 40 start does not repeat" (t_d1 0.31, t_d1_d2 0.08), (b) Fig 3 "40-yd start -> snaps" and "DL drill -> snaps = zero" panels, (c) the writeup's headline DL recommendation ("do not grade the pass-rush drill's first step"). The year-centering (`combine_player`) removes the mean shift between 2025 and 2023/24 but not the per-attempt jitter, and the 2025 DL class is measured on a different clock than 2023/24. Note `s_at_d1` (least onset-dependent metric) is exactly the one that transfers best for OL, consistent with an onset artifact rather than a football finding.
Fix options: flag attempts with first-frame s >= START_SPEED (or pre_s null) and exclude/sensitivity-test them; back-extrapolate onset from the first frames (s/a); or restrict transfer and reliability claims to onset-free metrics (speed/accel at fixed depth). At minimum, state the limitation in the writeup; currently it says "Movement onset = first frame at >=0.5 yd/s" as if it were always observed.

### C2. Writeup numbers not produced by the notebook / stale vs results.json
- `docs/writeup-same-yardstick.md:45` DL play-level model "31,806 snaps, p = 0.006 and 0.016" comes from `src/explore_dl_second_yard.py`, which is not in `SRC_FILES` (`build_final_notebook.py:17`) and not in `results.json`. Appendix claim "Numbers in this writeup are read from results.json" is false for this sentence.
- `writeup:49` "44,944 snaps" vs `results.json ol_model_n` = 45477.
- `writeup:18` "65 OL" vs `results.json n.ol_players` = 66 (65 is the count with a Combine drill; wording should say so).
- `writeup:59` "4-game rolling mean reaches ~0.7" vs `results.json weekly.games_for_r70` = 3 (minor, but the code chose 3 and the text says 4).

## High

### H1. Fig 2 compares non-comparable reliabilities
`make_figures.py:89-120`: in-game bars are Spearman-Brown corrected reliabilities of ~100-500-snap means; Combine bars are raw single-attempt correlations (labelled `r_sb` but not corrected). Title "Combine starts repeat far less" is then partly a statement about n per measurement, not about the drill. The quantity actually used downstream (mean of 2 attempts) has reliability 2r/(1+r) (e.g. smax 0.84, t_d1 0.48), and the comparable in-game quantity is single-snap reliability (will be far below 0.7). Report either per-measurement reliability on both sides or the reliability of the aggregate actually used.

### H2. Bootstrap CI ignores residualization uncertainty
`trench_analysis.py:101-109`: controls are residualized once, then rows of residuals are resampled. The CI treats the OLS nuisance coefficients as known; with n=53-69 and 3 controls this narrows CIs modestly. Re-residualize inside each bootstrap draw (cheap: 2000 x lstsq on 65 rows). Every CI in Fig 3/4/5 and `official` is affected (direction: too narrow).

### H3. z-scoring within position on tiny groups
`make_figures.py:66` z-scores all `g_` columns within `nfl_position`; group sizes in the analysis set: C=8, NT=3 (OL 29/29/8, DL 25/24/17/3). With NT=3, z is +-1 noise and `std` ddof=1 is unstable; a single NT drives tails of the DL correlations in Fig 5. Pool NT with DT (interior) and consider pooling C with G, or z within side only with position as a control.

## Medium

- M1 `make_figures.py:176-186` fig6: player means `pm` are computed on all snaps but the within-model sample `xx` is restricted (lined_up_position in T/G/C, non-null outcomes, players with Combine rows). `w_` deviations therefore do not have zero mean within player; the "within" model is not an exact fixed-effects estimator and the between/within decomposition is contaminated. Compute means on `x` itself, or use `C(nfl_id)` fixed effects. Also `cov_kwds groups` is built from a separately filtered frame; if any model variable (e.g. `combine_weight`) were NaN, statsmodels drops rows and the cluster vector misaligns silently or errors. Pass `groups` from the same frame the formula consumes after an explicit `dropna` on all model columns.
- M2 `make_figures.py:51-56` 40-yd `f_` z-scores are computed within draft year across all positions in `att` (WR/CB included), then merged to linemen. Per-year SD scaling thus depends on year-specific position mix; use linemen-only rows before z (as fig2 already does at line 98).
- M3 `trench_analysis.py:49-61` split-half pools 2023-2025 odd/even weeks per player, so "half a season per half" in the writeup is really "half of 1-3 seasons". Players in 3 seasons dominate the reliability. State n of seasons or compute within season.
- M4 `build_combine_trench_metrics.py:82-88` division by `norm` for non-primary drills with zero net displacement yields inf/NaN ux; harmless for the two primary drills (fixed axes) but will poison `set_angle` for other drills if used later.
- M5 `extract_body.py:83` `n_plays_all` triggers a second full scan of each multi-GB tracking CSV purely for a diagnostic; this alone likely doubles runtime and endangers the "~3 minutes" notebook claim (`build_final_notebook.py:30`). Compute from the already-collected frame or drop.
- M6 `trench_game_features.csv`: `t_max` min = 1.4 s and `n_frames` min = 18 on some plays, so `smax_15`, `t_d2`, `t_d1_d2` are computed on truncated windows without a flag. Filter `t_max >= 1.5` (or 2.0) in `load_game_features`.

## Low

- L1 `trench_features.py:81` comment "player's (offensive) left" is ambiguous; for DL it is the rusher's right. Document, since `side_10`/`side_game` are used for the side-split claim in the writeup.
- L2 `trench_analysis.py:67` `fisher_ci` unused (imports scipy lazily); remove.
- L3 `make_figures.py:113` `c40.rename(columns={"r_sb": "r_sb"})` no-op; the column name itself is misleading (see H1).
- L4 `make_figures.py:152,169` `np.polyfit` on columns that can contain NaN (g_t_d1_d2) silently returns NaN line coefficients; dropna before fitting.
- L5 `build_final_notebook.py:36-37` `Path.exists()` is False for a dangling symlink, so a rerun with a stale link hits `FileExistsError`; use `is_symlink() or exists()`.
- L6 `build_final_notebook.py:44` `r'''...'''` cells: assert covers `'''`, but a module ending in a backslash or containing `\'''`-adjacent sequences would break the literal; a `base64`/`json.dumps` payload is safer than raw-string embedding. Current files are fine.
- L7 `extract_body.py:50` `play_direction` null falls through to ux=+1 silently; assert non-null (currently 0 nulls).

## Reproducibility (d)
Notebook assembly is sound: `from trench_features import *` does export `pl`, `TOL`, `WINDOWS`; `ROOT = Path(__file__).resolve().parents[1]` resolves to `/kaggle/working` for written modules; the `data/raw` symlink satisfies `RAW` in both `build_combine_trench_metrics.py` and `make_figures.py`; outputs moved into `data/processed` before `make_figures.main()`. Risks: Kaggle pandas is 2.x (code is compatible), `explore_*` scripts are outside the notebook (C2), and runtime (M5).

## Recommended actions (priority)
1. C1: quantify and handle onset truncation; rerun Fig 2/3 and revise the DL drill conclusion and the "40 start does not repeat" claim, or scope claims to onset-free metrics.
2. C2: add the DL play-level model to `make_figures.py`/`results.json` (or drop the sentence); refresh 44,944 -> 45,477, 65 -> 66 (or say "65 with a drill"), 4-game -> 3-game or recompute.
3. H3: pool NT into interior before z-scoring; H2: bootstrap the residualization; H1: make Fig 2 comparable.
4. M1/M6: fix within-player means and truncated-window filter.

Status: DONE_WITH_CONCERNS
Summary: Coordinate conventions, orientation, bool parsing and the partial-correlation/Spearman-Brown mechanics are verified correct against the data; but Combine "onset" is truncated by recording start in 80-90% of 40 attempts and 58-78% of 2023/24 DL drill attempts, which undermines the DL-drill and 40-start findings, and several writeup numbers are not produced by (or disagree with) the notebook's results.json.
Concerns/Blockers: C1 may change the DL headline conclusion; C2 breaks the "all numbers from the notebook" claim.
