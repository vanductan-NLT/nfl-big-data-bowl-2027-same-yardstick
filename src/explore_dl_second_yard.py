"""Exploration (DL + OL): does the 40-yard dash start (official split and
tracking first second) predict in-game get-off / set, does in-game get-off
predict pressure after play-level controls, and how stable is it by week.

Usage: python src/explore_dl_second_yard.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trench_analysis import GAME_METRICS, corr_table, load_game_features, player_means, zscore_within  # noqa: E402

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
BURST = ["depth_05", "depth_10", "s_05", "s_10", "amax_05", "t_d1", "t_d2", "t_d1_d2", "s_at_d1", "s_at_d2", "smax_15"]


def combine_player(att: pd.DataFrame, drill: str, cols: list[str]) -> pd.DataFrame:
    """Mean over attempts of year-centred (z within draft_year) metrics."""
    d = att[att.drill_name == drill].copy()
    for c in cols:
        g = d.groupby("draft_year")[c]
        d[c] = (d[c] - g.transform("mean")) / g.transform("std")
    return d.groupby("nfl_id")[cols].mean().add_prefix(drill[:5].lower() + "_").reset_index()


def main() -> None:
    g = load_game_features(ROOT / "data/processed/trench_game_features.csv")
    att = pd.read_csv(ROOT / "data/processed/combine_trench_attempts.csv")
    players = pd.read_csv(ROOT / "data/raw/players.csv")
    comb = pd.read_csv(ROOT / "data/raw/combine_results.csv")
    base = players[["nfl_id", "nfl_position", "draft_year", "draft_overall_pick"]].merge(
        comb[["nfl_id", "combine_height", "combine_weight", "forty", "ten_yd_split", "ngs_athleticism_score"]], on="nfl_id")
    base["pick"] = base.draft_overall_pick.fillna(270)
    forty = combine_player(att, "FORTY_YARD_DASH", BURST)
    for side, drill, outs in [("DL", "PASS_RUSH_DRILL", ["pressured", "quick_pressure", "player_get_off"]),
                              ("OL", "PASS_RUSH_DROP", ["pressure_allowed", "peak_pressure_probability_allowed"])]:
        sub = g[g.side == side]
        drl = combine_player(att, drill, BURST)
        gm = player_means(sub, ["nfl_id"], GAME_METRICS + outs, 100).rename(columns={m: "g_" + m for m in GAME_METRICS + outs})
        d = base.merge(gm, on="nfl_id").merge(forty, on="nfl_id", how="left").merge(drl, on="nfl_id", how="left")
        d = zscore_within(d, [c for c in d.columns if c.startswith("g_")], ["nfl_position"])
        ys = ["g_t_d1", "g_t_d1_d2", "g_s_at_d1", "g_depth_10", "g_s_10"] + ["g_" + o for o in outs]
        xs = ["forty", "ten_yd_split", "ngs_athleticism_score", "combine_weight", "pick"] + [c for c in d.columns if c.startswith("forty_") or c.startswith(drill[:5].lower() + "_")]
        t = corr_table(d, xs, ys, controls=["combine_height", "combine_weight", "pick"], n_boot=500)
        print(f"=== {side}: Combine (official / 40 tracking / drill tracking) -> in-game, partial ht,wt,pick, n={len(d)}")
        print(t.pivot(index="x", columns="y", values="r").round(2).to_string())
        print("CI for ten_yd_split & forty_t_d1 / drill t_d1 vs g_t_d1:")
        print(t[(t.y == "g_t_d1") & (t.x.isin(["ten_yd_split", "forty_t_d1", "forty_s_at_d1", drill[:5].lower() + "_t_d1", drill[:5].lower() + "_s_at_d1"]))].round(2).to_string(index=False))
        # reliability of the Combine side: 40 attempt1 vs attempt2 for this group (year-centred)
        f2 = att[(att.drill_name == "FORTY_YARD_DASH") & att.nfl_id.isin(d.nfl_id)].copy()
        f2["rank"] = f2.groupby("nfl_id").attempt.rank(method="first")
        p1 = f2[f2["rank"] == 1].set_index("nfl_id"); p2 = f2[f2["rank"] == 2].set_index("nfl_id"); cm = p1.index.intersection(p2.index)
        rows = []
        for c in ["t_d1", "s_at_d1", "t_d1_d2", "s_10", "depth_10", "smax_15"]:
            a, b = p1.loc[cm, c].astype(float), p2.loc[cm, c].astype(float); yr = p1.loc[cm, "draft_year"]
            a = a - a.groupby(yr).transform("mean"); b = b - b.groupby(yr).transform("mean"); ok = a.notna() & b.notna()
            rows.append((c, int(ok.sum()), round(np.corrcoef(a[ok], b[ok])[0, 1], 2)))
        print(f"{side} 40-yd tracking attempt1 vs 2 (year-centred):", rows)

    # DL: in-game get-off -> pressure with play-level controls
    dl = g[g.side == "DL"].copy()
    pm = player_means(dl, ["nfl_id"], GAME_METRICS + ["player_get_off"], 100).set_index("nfl_id")
    x = dl[dl.nfl_id.isin(pm.index)].copy()
    for m in ["t_d1", "t_d1_d2", "s_at_d1", "depth_10", "player_get_off"]:
        x["P_" + m] = x.nfl_id.map(pm[m])
    x = x.merge(base[["nfl_id", "pick", "combine_weight"]], on="nfl_id")
    x["edge"] = x.lined_up_position.isin(["DE", "OLB"]).astype(float)
    x["blitz"] = x.blitzing.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).fillna(0.0)
    x = x.dropna(subset=["down", "yards_to_go"])
    x["down"] = x.down.astype(int)
    for f in ["pressured ~ edge + blitz + C(down) + yards_to_go + pick + combine_weight + P_player_get_off",
              "pressured ~ edge + blitz + C(down) + yards_to_go + pick + combine_weight + P_t_d1_d2",
              "pressured ~ edge + blitz + C(down) + yards_to_go + pick + combine_weight + P_s_at_d1",
              "pressured ~ edge + blitz + C(down) + yards_to_go + pick + combine_weight + P_player_get_off + P_t_d1_d2",
              "pressured ~ edge + blitz + C(down) + yards_to_go + pick + combine_weight + ten_yd_split"]:
        xx = x.merge(base[["nfl_id", "ten_yd_split"]], on="nfl_id") if "ten_yd_split" in f else x
        xx = xx.dropna(subset=[c for c in ["ten_yd_split"] if c in f])
        m = smf.ols(f, data=xx).fit(cov_type="cluster", cov_kwds={"groups": xx.nfl_id})
        keep = [i for i in m.params.index if i.startswith("P_") or i in ("ten_yd_split", "edge", "pick")]
        print("DL model:", f.split("~")[1].strip()[-45:], "n", int(m.nobs)); print(m.summary2().tables[1].loc[keep, ["Coef.", "Std.Err.", "P>|z|"]].round(4).to_string())
    # within-player play level: second yard -> pressure
    for m_ in ["t_d1", "t_d1_d2", "s_at_d1"]:
        x["w_" + m_] = x[m_] - x.groupby("nfl_id")[m_].transform("mean")
    xx = x.dropna(subset=["w_t_d1_d2"])
    m = smf.ols("pressured ~ w_t_d1 + w_t_d1_d2 + edge + blitz + C(down) + yards_to_go", data=xx).fit(cov_type="cluster", cov_kwds={"groups": xx.nfl_id})
    print("DL within-player:"); print(m.summary2().tables[1].loc[["w_t_d1", "w_t_d1_d2"], ["Coef.", "Std.Err.", "P>|z|"]].round(4).to_string())

    # week-level stability / trend (rookie season)
    dy = dl.nfl_id.map(players.set_index("nfl_id").draft_year)
    rk = dl[dl.season == dy]
    wk = rk.groupby(["nfl_id", "week"]).agg(t_d1_d2=("t_d1_d2", "mean"), s_at_d1=("s_at_d1", "mean"), n=("t_d1_d2", "size")).reset_index()
    wk = wk[wk.n >= 10]
    wk["t_d1_d2_c"] = wk.t_d1_d2 - wk.groupby("nfl_id").t_d1_d2.transform("mean")
    tr = wk.groupby("week").agg(t_d1_d2_c=("t_d1_d2_c", "mean"), n=("nfl_id", "nunique"))
    print("DL rookie-season week trend of (player-centred) 1->2 yd time:"); print(tr.round(4).T.to_string())
    print("per-player-game SD of t_d1_d2 (n>=10 plays):", round(wk.groupby("nfl_id").t_d1_d2.std().median(), 4), "between-player SD:", round(pm.t_d1_d2.std(), 4))


if __name__ == "__main__":
    main()
