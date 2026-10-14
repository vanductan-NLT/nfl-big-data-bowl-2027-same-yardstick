"""Exploration: OL side-matched Combine->game transfer, L/R asymmetry,
rookie-vs-later transfer, and play-level pressure models with controls.

Usage: python src/explore_ol_side_transfer.py
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
MET = ["depth_10", "depth_15", "s_10", "smax_10", "t_d1", "s_at_d1", "lat_abs_10", "set_angle_10"]
CTRL = ["combine_height", "combine_weight", "pick"]


def with_pick(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(players[["nfl_id", "draft_overall_pick"]].rename(columns={"draft_overall_pick": "dop"}), on="nfl_id")
    out["pick"] = out.dop.fillna(270)
    return out


def transfer_rows(d: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for m in MET:
        t = corr_table(d, [m + "_z"], [m + "_g"], controls=CTRL, n_boot=n_boot).iloc[0]
        rows.append((m, t.n, round(t.r, 2), round(t.lo, 2), round(t.hi, 2)))
    return pd.DataFrame(rows, columns=["metric", "n", "r_partial", "lo", "hi"])


def main() -> None:
    g = load_game_features(ROOT / "data/processed/trench_game_features.csv")
    cs = pd.read_csv(ROOT / "data/processed/combine_player_side.csv")
    players = pd.read_csv(ROOT / "data/raw/players.csv")
    ol = g[g.side == "OL"].copy()
    c = cs[cs.group == "OL"]

    # A. side-matched transfer (Combine L attempt vs in-game L-side sets, R vs R)
    gm = player_means(ol, ["nfl_id", "side_game"], GAME_METRICS, 50).rename(columns={"side_game": "side"})
    d = with_pick(c.merge(gm, on=["nfl_id", "side"], suffixes=("", "_g")), players)
    d = zscore_within(d, [m + "_g" for m in MET], ["nfl_position", "side"])
    print("A. side-matched rows", len(d), d.side.value_counts().to_dict())
    print(transfer_rows(d).to_string(index=False))
    d2 = with_pick(c.assign(side=c.side.map({"L": "R", "R": "L"})).merge(gm, on=["nfl_id", "side"], suffixes=("", "_g")), players)
    d2 = zscore_within(d2, [m + "_g" for m in MET], ["nfl_position", "side"])
    print("A2. side-MISMATCHED control"); print(transfer_rows(d2).to_string(index=False))

    # A3. asymmetry: Combine (L-R) vs in-game (L-R), and vs share of L-side sets
    w = c.pivot_table(index="nfl_id", columns="side", values=[m + "_z" for m in MET])
    gw = gm.pivot_table(index="nfl_id", columns="side", values=MET)
    share = ol.groupby("nfl_id").side_game.apply(lambda s: (s == "L").mean()).rename("L_share")
    npl = ol.groupby("nfl_id").size().rename("n")
    rows = []
    for m in MET:
        a = (w[(m + "_z", "L")] - w[(m + "_z", "R")]).rename("c")
        b = (gw[(m, "L")] - gw[(m, "R")]).rename("g")
        x = pd.concat([a, b, share, npl], axis=1).dropna(); x = x[x.n >= 100]
        rows.append((m, len(x), round(np.corrcoef(x.c, x.g)[0, 1], 2), round(np.corrcoef(x.c, x.L_share)[0, 1], 2)))
    print("A3. asymmetry Combine(L-R) vs game(L-R) | vs L-share")
    print(pd.DataFrame(rows, columns=["metric", "n", "r_asym", "r_Lshare"]).to_string(index=False))

    # B. transfer in rookie season vs later seasons
    c1 = c.groupby("nfl_id").agg({**{m + "_z": "mean" for m in MET}, "combine_height": "first", "combine_weight": "first",
                                  "draft_year": "first", "nfl_position": "first"}).reset_index()
    c1 = with_pick(c1, players)
    dy = ol.nfl_id.map(players.set_index("nfl_id").draft_year)
    for label, sub in [("rookie", ol[ol.season == dy]), ("later", ol[ol.season > dy])]:
        gm2 = player_means(sub, ["nfl_id"], MET, 100).rename(columns={m: m + "_g" for m in MET})
        dd = zscore_within(c1.merge(gm2, on="nfl_id"), [m + "_g" for m in MET], ["nfl_position"])
        print("B.", label); print(transfer_rows(dd, 500).to_string(index=False))

    # C. play-level pressure models with situational controls
    pm = player_means(ol, ["nfl_id"], GAME_METRICS, 100).set_index("nfl_id")
    x = ol[ol.nfl_id.isin(pm.index)].copy()
    for m in ["depth_10", "set_angle_10", "t_d1_d2", "s_at_d1", "lat_abs_10"]:
        x["P_" + m] = x.nfl_id.map(pm[m])
    x = with_pick(x, players).merge(c.drop_duplicates("nfl_id")[["nfl_id", "combine_weight"]], on="nfl_id")
    x = x[x.lined_up_position.isin(["T", "G", "C"])].dropna(subset=["peak_pressure_probability_allowed", "dropback_duration", "pass_rushers_encountered"])
    for yv in ["pressure_allowed", "peak_pressure_probability_allowed"]:
        f = f"{yv} ~ C(lined_up_position) + dropback_duration + pass_rushers_encountered + pick + combine_weight + P_depth_10 + P_set_angle_10 + P_t_d1_d2"
        m = smf.ols(f, data=x).fit(cov_type="cluster", cov_kwds={"groups": x.nfl_id})
        print("C.", yv, "n", int(m.nobs)); print(m.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(4).to_string())
    for m_ in ["depth_10", "set_angle_10", "t_d1_d2"]:
        x["w_" + m_] = x[m_] - x.groupby("nfl_id")[m_].transform("mean")
    xx = x.dropna(subset=["w_t_d1_d2"])
    f = "peak_pressure_probability_allowed ~ w_depth_10 + w_set_angle_10 + w_t_d1_d2 + dropback_duration + pass_rushers_encountered"
    m = smf.ols(f, data=xx).fit(cov_type="cluster", cov_kwds={"groups": xx.nfl_id})
    print("C2. within-player play-level"); print(m.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(4).to_string())


if __name__ == "__main__":
    main()
