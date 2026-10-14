# Same Yardstick: measuring trench movement identically at the Combine and after the snap

*Open Track. Companion notebook: [BDB27 same yardstick combine vs snap](https://www.kaggle.com/code/vanductan/bdb27-same-yardstick-combine-vs-snap). Code: [GitHub repository](https://github.com/vanductan-NLT/nfl-big-data-bowl-2027-same-yardstick).*

## The question

Offensive and defensive linemen run position drills at the Combine that look like their job: the OL kick-slide (`PASS_RUSH_DROP`) and the DL pass-rush drill (`PASS_RUSH_DRILL`). Scouts time them, but nobody measures them the way the same player is measured on Sundays. We built **one feature function** and applied it to both: to every tracked drill attempt, and to the first 1.5 seconds after `ball_snap` on every regular-season pass-pro snap (OL) and pass-rush snap (DL) of the 239 linemen in the cohort. If a movement metric means the same thing in both places, three questions that are usually mixed together can be asked separately:

1. **Does it repeat?** (attempt 1 vs 2 at the Combine; odd vs even weeks in games)
2. **Does it carry over?** (same metric, Combine → snaps, holding size and draft slot fixed)
3. **Does it matter?** (in-game metric → pressure, with situational controls)

Separating the links is the point. A drill can fail at link 2 while the in-game version of the same metric is excellent at link 3, and that tells an evaluator exactly where to look.

## Data and measurement

- **Combine side.** `combine_tracking.csv`, PLAYER rows. OL `PASS_RUSH_DROP` (239 attempts, 120 players) and DL `PASS_RUSH_DRILL` (225 attempts, 118 players), plus the 40 for comparison. Two things about this file shape everything downstream. First, the two attempts of each drill are **not replicates**: attempt 1 is run to one side and attempt 2 to the other (OL kick-slide left/right in 92–96% of attempts; DL rush arc likewise), so we treat them as a left and a right measurement and average them. Second, there is **no start signal**, and recording often begins after the player is already moving: the first recorded frame is at ≥0.5 yd/s in 86% of 40-yard attempts, 43% of DL drill attempts and 9% of OL drill attempts. We back-extrapolate the onset to rest assuming constant acceleration over the first frames (median shift 0.19–0.32 s) and exclude attempts whose first frame is already above 2 yd/s. Metrics that reference distance rather than the onset (speed when 1 yd is reached, time from 1 yd to 2 yd) are the least sensitive to this. The Combine `dis` column is not a per-frame step and is never used; distance comes from x/y.
- **Game side.** `game_tracking_2023–2025.csv`, regular season only, penalties and overtime removed, full 1.5 s window required. Snap time = first `ball_snap` frame; depth = yards moved toward the offensive backfield (`play_direction` sets the axis), lateral = yards across; orientation relative to the line of scrimmage from `o`. OL snaps are those with `dropback_duration` recorded (48,428 snaps), DL snaps those with `player_get_off` recorded (33,105). Analysis set: ≥100 snaps (66 OL, 69 DL; 65 OL also have the drill).
- **Metrics** (identical on both sides): depth at 0.5/1.0/1.5 s, lateral at 1.0 s, speed at 0.5/1.0 s, peak speed in 0–1.5 s, time to 1 yd and 2 yd, *time from 1 yd to 2 yd* ("the second yard"), speed when 1 yd is reached, set angle at 1.0 s. Our time-to-1-yd agrees with NGS `player_get_off` at r = 0.88 on 33,361 DL snaps, so the yardstick is sound.
- **Controls.** Combine metrics are centred within draft class (the three Combines were not recorded identically; the 40 is centred among linemen only), in-game metrics within position (T/G/C; DE/OLB/DT with NT pooled into DT). Correlations are partial on height, weight and overall draft slot (UDFA = 270); 95% CIs are bootstraps over players that refit the control regression in every resample. The primary transfer test was written down before running: depth at 1.0 s, Combine drill vs in-game.

**Figure 1** shows the yardstick: mean speed curves at the Combine and after the snap. The OL drill peaks and decays (a kick-slide then a hitch); a real pass set keeps accelerating. The DL drill is a faster version of a real get-off. Neither looks like the other at the level of curves, which is exactly why metric-by-metric comparison is needed.

![Figure 1](https://raw.githubusercontent.com/vanductan-NLT/nfl-big-data-bowl-2027-same-yardstick/master/output/figures/fig1_profiles.png)

## Link 1 — what repeats

**Figure 2.** In-game trench movement is a fingerprint: splitting each player's 2023–2025 regular-season snaps into odd and even weeks, Spearman-Brown reliability is 0.94–0.97 for OL set metrics and 0.92–0.98 for DL get-off metrics. The outcomes are noisier: pressure allowed 0.61, peak pressure probability allowed 0.85; DL pressure rate 0.69; NGS get-off 0.95. At the Combine the picture is different. Because the position drills are run once per side, we use the 40 (168 linemen with two attempts) and report the reliability of the two-attempt mean, which is what any analysis uses: peak speed 0.74, speed at 1.0 s 0.67, depth at 1.0 s 0.61, time to 1 yd 0.56, speed at 1 yd 0.49, and the second yard only 0.12. A start measured twice is a noisy number; the same start measured over 400 snaps is not.

![Figure 2](https://raw.githubusercontent.com/vanductan-NLT/nfl-big-data-bowl-2027-same-yardstick/master/output/figures/fig2_reliability.png)

## Link 2 — what carries over

**Figure 3** gives the same-metric transfer for four Combine sources.

- **OL kick-slide drill → OL snaps.** The pre-specified primary, depth at 1.0 s, is r = 0.14 (−0.07 to 0.35). The metric least exposed to onset error, speed when 1 yd is reached, is the strongest at **r = 0.44 (0.12 to 0.65)** (Figure 4; it is the best of six, chosen after seeing the data, and should be read that way). The 40-yard start carries nothing useful for the set (its speed at 1 yd has the wrong sign), and neither do the official 40 or 10-yard split (|r| ≤ 0.19). The position drill contains information about how a prospect will set that the sprint does not.
- **DL pass-rush drill → DL snaps.** Nothing clears zero: all six metrics between −0.02 and 0.14 with CIs of about ±0.25. The 40-yard start does no better (−0.13 to 0.14). Among official numbers, the 10-yard split relates to in-game NGS get-off at 0.25 (−0.07 to 0.51) and to pressure rate at −0.20 (−0.42 to 0.03). Draft slot is the strongest Combine-day predictor of in-game pressure rate (−0.33, −0.52 to −0.09): teams already price what the drill cannot see.

![Figure 3](https://raw.githubusercontent.com/vanductan-NLT/nfl-big-data-bowl-2027-same-yardstick/master/output/figures/fig3_transfer.png)

![Figure 4](https://raw.githubusercontent.com/vanductan-NLT/nfl-big-data-bowl-2027-same-yardstick/master/output/figures/fig4_ol_scatter.png)

## Link 3 — what matters

**DL (Figure 5).** The in-game get-off predicts pressure, and the second yard predicts it better than the first. With weight and draft slot held fixed, NGS get-off correlates −0.44 (−0.59 to −0.25) with a player's pressure rate; time from 1 yd to 2 yd correlates **−0.51 (−0.65 to −0.34)**; speed at 1 yd +0.50 (0.31 to 0.65). In a play-level model with edge/interior alignment, blitz, down, distance, draft slot and weight (31,803 snaps, SEs clustered by player), get-off and the second yard are each significant when entered together (p = 0.006 and 0.016): they are not the same thing. The official 10-yard split, in the same model, is a weaker predictor (p = 0.02, 24,558 snaps with a split). The second yard is where the rusher has already won or lost the hands, and it does not depend on where the tracking's clock starts.

![Figure 5](https://raw.githubusercontent.com/vanductan-NLT/nfl-big-data-bowl-2027-same-yardstick/master/output/figures/fig5_dl_outcome.png)

**OL (Figure 6).** Set style is not a grade. Between players, mean set depth at 1.0 s has no relation to peak pressure probability allowed once position, dropback length and rushers faced are controlled (coefficient p = 0.87; 38,621 snaps, SEs clustered by player); set angle p = 0.40; pace of retreat is borderline (slower from 1 to 2 yd of depth → less pressure, p = 0.07). Within a player the sign is unambiguous: the snaps where he sets deeper than his own average are the pressured ones (+0.059 pressure probability per extra yard at 1.0 s, p < 0.001). Retreat is reactive: a lineman being driven back goes deeper. So the kick-slide drill tells you how a prospect moves, and that does carry to the NFL (link 2), but "deeper and faster" is a style, not quality.

![Figure 6](https://raw.githubusercontent.com/vanductan-NLT/nfl-big-data-bowl-2027-same-yardstick/master/output/figures/fig6_ol_within.png)

## What a team can do with this

**At the Combine.**
- *OL*: use the kick-slide drill as a **movement-profile projection**, not a pass-pro grade. Speed at 1 yd predicts the set a prospect will show on NFL snaps (r ≈ 0.4 beyond size and slot; depth and timing metrics ≈ 0.15). That is useful for scheme fit — vertical-set systems want depth early; jump-set systems do not — and for coaching plans, because the in-game set is highly stable and will not change by accident.
- *DL*: do **not** grade the pass-rush drill's first step. Even after correcting the start, it does not carry to NFL get-off (CI ±0.25), and the 40's first yard does no better. The official 10-yard split carries a little; draft slot carries more. Spend the Combine budget on the parts of the drill that are not the start (hand usage, bend) or on film.

**Week to week.** The in-game metrics are the actionable ones. A rusher's second-yard time is stable across halves of a season and predicts pressure beyond size and slot; a lineman's set signature is stable to 0.95. **Figure 7** shows the practical rule for monitoring: one game is as noisy as the between-player spread (game SD 0.055 s vs between-player SD 0.053 s), so a weekly monitor should use a 3–4 game rolling mean, which reaches reliability ≈ 0.7. Three rookies are shown across 2023–2025; swings of a full between-player SD over a month are visible and are the kind of signal a coaching staff can check against injury reports and opponent quality. The same chart for OL set depth flags linemen whose sets are drifting deeper, which, by the within-player result, is a lineman being beaten more often, not a lineman choosing a new style.

![Figure 7](https://raw.githubusercontent.com/vanductan-NLT/nfl-big-data-bowl-2027-same-yardstick/master/output/figures/fig7_weekly.png)

## Limits

- 65–69 players per group, three draft classes. A single pre-chosen test needs |r| ≈ 0.24 to clear p < 0.05 at this n; the pre-specified OL test did not clear it, and the 0.44 is the best of six. Replicate on the 2026 class before treating it as the number.
- The Combine file has no start signal; our onset correction is a model, and onset-dependent Combine metrics (time to 1 yd, depth at 1.0 s) carry residual error that pushes their transfer toward zero. The distance-referenced metrics are the ones to trust on the Combine side.
- Position drills were run once per side; we cannot measure their repeatability directly and do not claim it.
- Pressure outcomes are team-dependent (scheme, matchup, QB). Our controls are situational, not matchup-level; the DL cohort's pressure-rate reliability is 0.69, which caps any correlation with it near 0.8.
- The 2025 class has one season; career exposure differs by class, so we use per-snap rates and never career totals.
- Defensive backs and receivers have no symmetric in-game outcome in `player_play`; the yardstick applies, the outcome link does not.

## Appendix

All code is in the companion notebook (one feature module, extraction, Combine metrics, analysis, figures; about 3 minutes on CPU) and in the GitHub repository it links to. Every number in this writeup is read from the `results.json` that the notebook writes.
