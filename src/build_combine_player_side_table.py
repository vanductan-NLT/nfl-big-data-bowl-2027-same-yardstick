"""Per-player, per-side Combine table for the two trench drills.

Both drills are run once to each side (OL kick-slide left/right, DL rush arc
left/right), so attempts are NOT replicates: keep one row per player x side
and also a player-level mean. Metrics are centred within draft_year x
combine_position because the three Combines were not recorded identically.

Usage: python src/build_combine_player_side_table.py
Writes data/processed/combine_player_side.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

PRIMARY = {"OL": "PASS_RUSH_DROP", "DL": "PASS_RUSH_DRILL"}
METRICS = [
    "depth_05", "depth_10", "depth_15", "depth_20", "lat_abs_10", "lat_abs_15", "s_05", "s_10",
    "s_15", "smax_10", "smax_15", "amax_05", "amax_10", "t_s1", "t_s2", "t_s3", "t_d1", "t_d2",
    "t_d1_d2", "s_at_d1", "s_at_d2", "set_angle_10", "smax_all", "duration",
    "peak1_t", "peak1_s", "dip_t", "dip_s", "dip_depth", "redirect_t",
]


def main() -> None:
    att = pd.read_csv(PROC / "combine_trench_attempts.csv")
    players = pd.read_csv(RAW / "players.csv")
    comb = pd.read_csv(RAW / "combine_results.csv")
    att = att.merge(players[["nfl_id", "nfl_position", "draft_overall_pick", "draft_round"]], on="nfl_id")
    att = att.merge(comb[["nfl_id", "combine_position", "combine_height", "combine_weight", "arm_length",
                          "forty", "ten_yd_split", "three_cone", "short_shuttle", "ngs_athleticism_score"]], on="nfl_id")
    att["side"] = att["side_10"]
    rows = []
    for grp, drill in PRIMARY.items():
        d = att[att.drill_name == drill].copy()
        d["group"] = grp
        # one row per player x side (first attempt on that side)
        d = d.sort_values(["nfl_id", "side", "attempt"]).drop_duplicates(["nfl_id", "side"])
        rows.append(d)
    ps = pd.concat(rows, ignore_index=True)
    keep = ["nfl_id", "group", "side", "draft_year", "drill_name", "attempt", "nfl_position", "combine_position",
            "draft_overall_pick", "draft_round", "combine_height", "combine_weight", "arm_length", "forty",
            "ten_yd_split", "three_cone", "short_shuttle", "ngs_athleticism_score"] + [m for m in METRICS if m in ps.columns]
    ps = ps[keep]
    # year x group centred versions (z within draft_year x group)
    for m in METRICS:
        if m not in ps.columns:
            continue
        g = ps.groupby(["draft_year", "group"])[m]
        ps[m + "_z"] = (ps[m] - g.transform("mean")) / g.transform("std")
    ps.to_csv(PROC / "combine_player_side.csv", index=False)
    print(ps.groupby(["group", "side"]).size())
    print(ps.groupby("group").nfl_id.nunique())


if __name__ == "__main__":
    main()
