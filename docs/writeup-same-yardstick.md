# Same Yardstick: measuring trench movement identically at the Combine and after the snap

*Open Track. Companion notebook: [Same Yardstick: Combine vs snap trench movement](https://www.kaggle.com/code/vanductan/bdb27-same-yardstick-combine-vs-snap). Code: GitHub repo linked in the notebook.*

## The question

Offensive and defensive linemen run position drills at the Combine that look like their job: the OL kick-slide (`PASS_RUSH_DROP`) and the DL pass-rush drill (`PASS_RUSH_DRILL`). Scouts time them, but nobody measures them the way the same player is measured on Sundays. We built **one feature function** and applied it to both: to every tracked drill attempt, and to the first 1.5 seconds after `ball_snap` on every regular-season pass-pro snap (OL) and pass-rush snap (DL) of the 239 linemen in the cohort. If a movement metric means the same thing in both places, we can ask three separate questions that are usually mixed together:

1. **Does it repeat?** (attempt 1 vs 2 at the Combine; odd vs even weeks in games)
2. **Does it carry over?** (same metric, Combine → snaps, holding size and draft slot fixed)
3. **Does it matter?** (in-game metric → pressure, with situational controls)

Separating the links is the point. A drill can fail at link 2 while the in-game version of the same metric is excellent at link 3, and that tells an evaluator exactly where to look.

## Data and measurement

- **Combine side.** `combine_tracking.csv`, PLAYER rows. OL `PASS_RUSH_DROP` (239 attempts, 120 players) and DL `PASS_RUSH_DRILL` (225 attempts, 118 players), plus the 40 for comparison. Movement onset = first frame at ≥0.5 yd/s. Important discovery: the two attempts of each drill are **not replicates**. Attempt 1 is run to one side, attempt 2 to the other (OL kick-slide left/right: 92–96% of attempts split by sign of lateral movement; DL rush arc likewise). We treat them as a left-side and right-side measurement and average them. The Combine `dis` column is not a per-frame step and is never used; distance comes from x/y.
- **Game side.** `game_tracking_2023–2025.csv`, regular season only, penalties and overtime removed. Snap time = first `ball_snap` frame; depth = yards moved toward the offensive backfield (`play_direction` sets the axis), lateral = yards across; orientation relative to the line of scrimmage from `o`. OL snaps are those with `dropback_duration` recorded (48,671 snaps, 89 players); DL snaps those with `player_get_off` recorded (33,373 snaps, 98 players). Analysis set: ≥100 snaps (65 OL, 69 DL).
- **Metrics** (identical on both sides): depth at 0.5/1.0/1.5 s, lateral at 1.0 s, speed at 0.5/1.0 s, peak speed in 0–1.5 s, time to 1 yd and 2 yd, *time from 1 yd to 2 yd* ("the second yard", onset-free), speed when 1 yd is reached, set angle at 1.0 s. Our time-to-1-yd agrees with NGS `player_get_off` at r = 0.88 on 33,361 DL snaps, so the yardstick is sound.
- **Controls.** All Combine metrics are centred within draft class (the three Combines were not recorded identically), in-game metrics within position (T/G/C; DE/OLB/DT/NT). Correlations are partial on height, weight and overall draft slot (UDFA = 270), with 95% bootstrap CIs over players. The primary transfer test was written down before running: depth at 1.0 s, Combine drill vs in-game.

**Figure 1** shows the yardstick: mean speed curves at the Combine and after the snap. The OL drill peaks and decays (a kick-slide then a hitch); a real pass set keeps accelerating. The DL drill is a faster version of a real get-off. Neither looks like the other at the level of curves, which is exactly why metric-by-metric comparison is needed.

![Figure 1](fig1_profiles.png)

## Link 1 — what repeats

**Figure 2.** In-game trench movement is a fingerprint: with half a season per half, odd-vs-even-week reliability is 0.94–0.98 for OL set metrics and 0.92–0.98 for DL get-off metrics (Spearman-Brown). The outcomes are noisier: pressure allowed 0.61, peak pressure probability allowed 0.85; DL pressure rate 0.69; NGS get-off 0.95. At the Combine the picture is different. Because the position drills are run once per side, we use the 40 (168 linemen with two attempts): peak speed repeats (0.72), speed at 1.0 s moderately (0.60), but the start does not — time to 1 yd 0.31, second yard 0.08. A Combine start measured once is a noisy number; the same start measured over 400 snaps is not.

![Figure 2](fig2_reliability.png)

## Link 2 — what carries over

**Figure 3** gives the same-metric transfer for four Combine sources.

- **OL kick-slide drill → OL snaps.** The pre-specified primary, depth at 1.0 s, is r = 0.21 (−0.01 to 0.43). Three of the six metrics sit around 0.2; speed when 1 yd is reached is the strongest at **r = 0.40 (0.09 to 0.62)** (Figure 4; chosen after seeing the data, so read it as the best of six, not a pre-stated test). The 40-yard start carries nothing useful for the set (speed at 1 yd even has the wrong sign), and neither do the official 40 or 10-yard split (|r| ≤ 0.19). The position drill contains information about how a prospect will set that the sprint does not.
- **DL pass-rush drill → DL snaps.** Essentially zero: depth, speed, time to 1 yd all between 0.03 and 0.08 with CIs of ±0.25. The second yard is the only metric that edges above zero (0.23, 0.00 to 0.42). The 40-yard start does slightly better for one metric (speed at 1 yd 0.31, 0.06 to 0.55), and the official 10-yard split relates to in-game NGS get-off at 0.25 (−0.03 to 0.51). Draft slot is the strongest Combine-day predictor of in-game pressure rate (−0.33, −0.52 to −0.09): teams already price what the drill cannot see.

![Figure 3](fig3_transfer.png)

![Figure 4](fig4_ol_scatter.png)

## Link 3 — what matters

**DL (Figure 5).** The in-game get-off predicts pressure, and the second yard predicts it better than the first. With weight and draft slot held fixed, NGS get-off correlates −0.44 (−0.59 to −0.26) with a player's pressure rate; time from 1 yd to 2 yd correlates **−0.51 (−0.65 to −0.34)**; speed at 1 yd +0.50 (0.32 to 0.65). In a play-level model with edge/interior alignment, blitz, down, distance, draft slot and weight (31,806 snaps, SEs clustered by player), get-off and the second yard are each significant when entered together (p = 0.006 and 0.016): they are not the same thing. The second yard is where the rusher has already won or lost the hands; it is also onset-free, so it does not depend on the snap-timing of the tracking.

![Figure 5](fig5_dl_outcome.png)

**OL (Figure 6).** Set style is not a grade. Between players, mean set depth at 1.0 s has no relation to peak pressure probability allowed once position, dropback length and rushers faced are controlled (coefficient p = 0.91, 44,944 snaps); set angle p = 0.16. The one set metric that survives is the pace of retreat: linemen who take longer from 1 yd to 2 yd of depth allow *less* pressure (−0.21 per second, p = 0.02). Within a player the sign is unambiguous: the snaps where he sets deeper than his own average are the pressured ones (+0.063 pressure probability per extra yard at 1.0 s, p < 0.001). Retreat is reactive: a lineman being driven back goes deeper. So the kick-slide drill tells you how a prospect moves, and that does carry to the NFL (link 2), but "deeper and faster" is a style, not quality.

![Figure 6](fig6_ol_within.png)

## What a team can do with this

**At the Combine.**
- *OL*: use the kick-slide drill as a **movement-profile projection**, not a pass-pro grade. Speed at 1 yd and depth at 1.0 s predict the set a prospect will show on NFL snaps (r ≈ 0.2–0.4 beyond size and slot). That is useful for scheme fit — vertical-set systems want depth early; jump-set systems do not — and for coaching plans, because the in-game set is highly stable and will not change by accident.
- *DL*: do **not** grade the pass-rush drill's first step. Measured once, it does not repeat, and it does not carry to NFL get-off (CI ±0.25). The 10-yard split and the 40's first yard carry a little; draft slot carries more. Spend the Combine budget on the parts of the drill that are not the start (hand usage, bend) or on film.

**Week to week.** The in-game metrics are the actionable ones. A rusher's second-yard time is stable across halves of a season and predicts pressure beyond size and slot; a lineman's set signature is stable to 0.95. **Figure 7** shows the practical rule for monitoring: one game is as noisy as the between-player spread (game SD 0.055 s vs between-player SD 0.053 s), so a weekly monitor should use a 4-game rolling mean, which reaches reliability ≈ 0.7. Three rookies are shown across 2023–2025; swings of a full between-player SD over a month are visible and are the kind of signal a coaching staff can check against injury reports and opponent quality. The same chart for OL set depth flags linemen whose sets are drifting deeper, which, by the within-player result, is a lineman being beaten more often, not a lineman choosing a new style.

![Figure 7](fig7_weekly.png)

## Limits

- 65–69 players per group, three draft classes. A single pre-chosen test needs |r| ≈ 0.24 to clear p < 0.05 at this n, and the OL "speed at 1 yd" result is the best of six; replicate on the 2026 class before treating 0.40 as the number.
- Position drills were run once per side; we cannot measure their repeatability directly and do not claim it.
- Pressure outcomes are team-dependent (scheme, matchup, QB). Our controls are situational, not matchup-level; the DL cohort's pressure rate reliability is 0.69, which caps any correlation with it near 0.8.
- The 2025 class has one season; career exposure differs by class, so we use per-snap rates and never career totals.
- Defensive backs and receivers have no symmetric in-game outcome in `player_play`; the yardstick applies, the outcome link does not.

## Appendix

All code is in the companion notebook (one feature module, extraction, Combine metrics, analysis, figures; ~3 minutes on CPU) and in the GitHub repository it links to. Numbers in this writeup are read from `results.json` produced by that notebook.
