"""Compute the writeup numbers and draw the figures.

Runs locally (paths under data/processed) and inside the Kaggle notebook
(paths under /kaggle/working) - set PROC / OUT accordingly.

Usage: python src/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trench_analysis import (GAME_METRICS, corr_table, load_game_features, player_means,  # noqa: E402
                             split_half_reliability, zscore_within)

PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_COMBINE, C_GAME, C_OL, C_DL = "#d95f02", "#1b9e77", "#2c7fb8", "#c51b7d"
LABELS = {"depth_10": "depth at 1.0 s (yd)", "s_10": "speed at 1.0 s", "t_d1": "time to 1 yd", "s_at_d1": "speed at 1 yd",
          "t_d1_d2": "time 1→2 yd", "smax_15": "peak speed 0–1.5 s", "set_angle_10": "set angle at 1.0 s", "orel_10": "hip angle at 1.0 s",
          "lat_abs_10": "lateral at 1.0 s", "player_get_off": "NGS get-off", "pressure_allowed": "pressure allowed",
          "peak_pressure_probability_allowed": "peak pressure prob. allowed", "pressured": "pressure rate", "depth_05": "depth at 0.5 s",
          "s_05": "speed at 0.5 s", "amax_05": "peak accel 0–0.5 s"}
MET = ["depth_10", "s_10", "t_d1", "s_at_d1", "t_d1_d2", "smax_15", "set_angle_10", "lat_abs_10"]
CTRL = ["combine_height", "combine_weight", "pick"]
RES: dict = {}


def load_all():
    g = load_game_features(PROC / "trench_game_features.csv")
    att = pd.read_csv(PROC / "combine_trench_attempts.csv")
    players = pd.read_csv(RAW / "players.csv")
    comb = pd.read_csv(RAW / "combine_results.csv")
    base = players[["nfl_id", "display_name", "nfl_position", "draft_year", "draft_overall_pick"]].merge(
        comb[["nfl_id", "combine_height", "combine_weight", "forty", "ten_yd_split"]], on="nfl_id")
    base["pick"] = base.draft_overall_pick.fillna(270)
    return g, att, players, base


def combine_player(att, drill, cols, ids=None):
    """Mean over attempts of year-centred (z within draft_year) metrics; `ids`
    restricts the centring population (e.g. linemen only for the 40)."""
    d = att[att.drill_name == drill].copy()
    if ids is not None:
        d = d[d.nfl_id.isin(ids)]
    for c in cols:
        grp = d.groupby("draft_year")[c]
        d[c] = (d[c] - grp.transform("mean")) / grp.transform("std")
    return d.groupby("nfl_id")[cols].mean().reset_index()


def player_table(g, att, base, side, drill, outs, min_plays=100):
    """One row per player: Combine drill z-metrics (c_), 40 z-metrics (f_), in-game means (g_) z within position."""
    gm = player_means(g[g.side == side], ["nfl_id"], GAME_METRICS + outs, min_plays)
    gm = gm.rename(columns={m: "g_" + m for m in GAME_METRICS + outs})
    ids = base[base.nfl_position.isin(["T", "G", "C"] if side == "OL" else ["DT", "DE", "OLB", "NT"])].nfl_id
    c = combine_player(att, drill, MET).rename(columns={m: "c_" + m for m in MET})
    f = combine_player(att, "FORTY_YARD_DASH", MET, ids).rename(columns={m: "f_" + m for m in MET})
    d = base.merge(gm, on="nfl_id").merge(c, on="nfl_id", how="left").merge(f, on="nfl_id", how="left")
    return zscore_within(d, [x for x in d.columns if x.startswith("g_")], ["nfl_position"])


# ----------------------------------------------------------------------------- figures
def fig1_profiles():
    """Mean speed profile 0-1.5 s: Combine drill vs in-game, OL and DL."""
    cp = pd.read_parquet(PROC / "combine_trench_profiles.parquet")
    gp = pd.read_parquet(PROC / "trench_game_profiles.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, side, drill, title in [(axes[0], "OL", "PASS_RUSH_DROP", "Offensive line: kick-slide drill vs pass-pro snap"),
                                   (axes[1], "DL", "PASS_RUSH_DRILL", "Defensive line: pass-rush drill vs pass-rush snap")]:
        for df, lab, col in [(cp[cp.drill_name == drill], "Combine drill", C_COMBINE), (gp[gp.side == side], "NFL snaps (REG)", C_GAME)]:
            t = np.round(np.concatenate(df.t_list.values), 1); s = np.concatenate(df.s_list.values)
            ok = (t >= 0) & (t <= 1.5)
            prof = pd.DataFrame({"t": t[ok], "s": s[ok]}).groupby("t").s.agg(["mean", lambda v: v.quantile(.25), lambda v: v.quantile(.75)])
            ax.plot(prof.index, prof["mean"], color=col, lw=2.2, label=f"{lab} (n={len(df):,})")
            ax.fill_between(prof.index, prof.iloc[:, 1], prof.iloc[:, 2], color=col, alpha=.15)
        ax.set_title(title, fontsize=10); ax.set_xlabel("seconds after movement start / ball snap"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("speed (yd/s), mean and IQR")
    fig.suptitle("Figure 1. One yardstick: the same speed-vs-time measurement at the Combine and after the snap", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig1_profiles.png", dpi=160); plt.close(fig)


def fig2_reliability(g, att):
    rows = []
    for side, outs in [("OL", ["pressure_allowed", "peak_pressure_probability_allowed"]), ("DL", ["player_get_off", "pressured"])]:
        r = split_half_reliability(g[g.side == side], ["nfl_id"], MET + outs, 100)
        r["side"] = side; rows.append(r)
    rel = pd.concat(rows)
    # Combine 40: attempt 1 vs 2, year-centred, linemen only
    f = att[(att.drill_name == "FORTY_YARD_DASH")].copy()
    pos = pd.read_csv(RAW / "players.csv").set_index("nfl_id").nfl_position
    f = f[f.nfl_id.map(pos).isin(["T", "G", "C", "DT", "DE", "OLB", "NT"])]
    f["rank"] = f.groupby("nfl_id").attempt.rank(method="first")
    p1 = f[f["rank"] == 1].set_index("nfl_id"); p2 = f[f["rank"] == 2].set_index("nfl_id"); cm = p1.index.intersection(p2.index)
    c40 = []
    for m in MET:
        a, b = p1.loc[cm, m].astype(float), p2.loc[cm, m].astype(float); yr = p1.loc[cm, "draft_year"]
        a = a - a.groupby(yr).transform("mean"); b = b - b.groupby(yr).transform("mean"); ok = a.notna() & b.notna()
        r1 = np.corrcoef(a[ok], b[ok])[0, 1]
        # r_sb = reliability of the mean of the two attempts (the quantity used downstream)
        c40.append({"metric": m, "n": int(ok.sum()), "r_single": r1, "r_sb": 2 * r1 / (1 + r1)})
    c40 = pd.DataFrame(c40)
    RES["reliability_game"] = rel.round(3).to_dict("records"); RES["reliability_40"] = c40.round(3).to_dict("records")
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ms = MET
    y = np.arange(len(ms))
    for off, (lab, df, col) in enumerate([("NFL snaps, OL (odd vs even weeks, n=%d)" % rel[rel.side == "OL"].n.max(), rel[rel.side == "OL"], C_OL),
                                          ("NFL snaps, DL (odd vs even weeks, n=%d)" % rel[rel.side == "DL"].n.max(), rel[rel.side == "DL"], C_DL),
                                          ("Combine 40, linemen, mean of 2 attempts (SB from attempt 1 vs 2, n=%d)" % c40.n.max(), c40, C_COMBINE)]):
        vals = [df.set_index("metric").loc[m, "r_sb"] for m in ms]
        ax.barh(y + (off - 1) * 0.27, vals, height=0.25, color=col, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[m] for m in ms]); ax.invert_yaxis(); ax.set_xlim(-0.05, 1.05)
    ax.axvline(0.7, ls="--", color="grey", lw=1); ax.text(0.71, -0.45, "0.70", color="grey", fontsize=8)
    ax.set_xlabel("reliability of the quantity used (split-half with Spearman-Brown for snaps; two-attempt mean for the Combine 40)", fontsize=8.5)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False); ax.grid(axis="x", alpha=.3)
    ax.set_title("Figure 2. In-game trench movement is a stable signature; Combine starts repeat far less", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig2_reliability.png", dpi=160); plt.close(fig)


def fig3_transfer(dol, ddl):
    """Forest plot: same metric Combine -> in-game, partial on height, weight, pick."""
    panels = []
    for name, d, src in [("OL kick-slide drill → OL snaps", dol, "c_"), ("OL 40-yd start → OL snaps", dol, "f_"),
                         ("DL pass-rush drill → DL snaps", ddl, "c_"), ("DL 40-yd start → DL snaps", ddl, "f_")]:
        t = pd.concat([corr_table(d, [src + m], ["g_" + m], controls=CTRL, n_boot=2000, seed=1) for m in MET[:6]])
        t["metric"] = MET[:6]; t["panel"] = name; panels.append(t)
    t = pd.concat(panels)
    RES["transfer"] = t.round(3).to_dict("records")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for ax, (name, sub) in zip(axes, t.groupby("panel", sort=False)):
        y = np.arange(len(sub)); col = C_OL if name.startswith("OL") else C_DL
        ax.errorbar(sub.r, y, xerr=[sub.r - sub.lo, sub.hi - sub.r], fmt="o", color=col, capsize=3)
        ax.axvline(0, color="grey", lw=1); ax.set_xlim(-0.6, 0.8); ax.set_title(f"{name}\n(n={int(sub.n.max())})", fontsize=9); ax.grid(axis="x", alpha=.3)
        ax.set_yticks(y); ax.set_yticklabels([LABELS[m] for m in sub.metric])
    axes[0].invert_yaxis(); fig.supxlabel("partial correlation (height, weight, draft slot held fixed), 95% bootstrap CI", fontsize=9)
    fig.suptitle("Figure 3. Does the same movement metric carry from the Combine to NFL snaps?", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig3_transfer.png", dpi=160); plt.close(fig)


def fig4_ol_scatter(dol):
    d = dol.dropna(subset=["c_s_at_d1", "g_s_at_d1"])
    t = corr_table(d, ["c_s_at_d1"], ["g_s_at_d1"], controls=CTRL, n_boot=2000, seed=2).iloc[0]
    RES["ol_scatter"] = {"n": int(t.n), "r": round(t.r, 3), "lo": round(t.lo, 3), "hi": round(t.hi, 3)}
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for pos, mk in [("T", "o"), ("G", "s"), ("C", "^")]:
        s = d[d.nfl_position == pos]; ax.scatter(s.c_s_at_d1, s.g_s_at_d1, marker=mk, s=42, color=C_OL, alpha=.75, label=f"{pos} (n={len(s)})")
    for _, r in pd.concat([d.nlargest(2, "g_s_at_d1"), d.nsmallest(2, "g_s_at_d1")]).iterrows():
        ax.annotate(r.display_name, (r.c_s_at_d1, r.g_s_at_d1), fontsize=7, xytext=(4, 3), textcoords="offset points")
    b = np.polyfit(d.c_s_at_d1, d.g_s_at_d1, 1); xs = np.linspace(d.c_s_at_d1.min(), d.c_s_at_d1.max(), 10); ax.plot(xs, np.polyval(b, xs), color="k", lw=1)
    ax.set_xlabel("Combine kick-slide drill: speed at 1 yd (z within class, mean of L/R)"); ax.set_ylabel("NFL pass-pro snaps: speed at 1 yd (z within position)")
    ax.set_title(f"Figure 4. OL: the kick-slide drill predicts the in-game set\npartial r = {t.r:.2f} (95% CI {t.lo:.2f} to {t.hi:.2f}), n = {int(t.n)}", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "fig4_ol_scatter.png", dpi=160); plt.close(fig)


def fig5_dl_outcome(ddl):
    d = ddl.copy(); d["edge"] = d.nfl_position.isin(["DE", "OLB"])
    tg = corr_table(d, ["g_player_get_off"], ["g_pressured"], controls=["combine_weight", "pick"], n_boot=2000, seed=3).iloc[0]
    t2 = corr_table(d, ["g_t_d1_d2"], ["g_pressured"], controls=["combine_weight", "pick"], n_boot=2000, seed=3).iloc[0]
    ts = corr_table(d, ["g_s_at_d1"], ["g_pressured"], controls=["combine_weight", "pick"], n_boot=2000, seed=3).iloc[0]
    RES["dl_outcome"] = {k: {"n": int(v.n), "r": round(v.r, 3), "lo": round(v.lo, 3), "hi": round(v.hi, 3)} for k, v in [("get_off", tg), ("t_d1_d2", t2), ("s_at_d1", ts)]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, xcol, tt, lab in [(axes[0], "g_player_get_off", tg, "NGS get-off: time to cross the line (z within position)"),
                              (axes[1], "g_t_d1_d2", t2, "second yard: time from 1 yd to 2 yd of penetration (z within position)")]:
        for e, mk, nm in [(True, "o", "edge (DE/OLB)"), (False, "s", "interior (DT/NT)")]:
            s = d[d.edge == e]; ax.scatter(s[xcol], s.g_pressured, marker=mk, s=42, color=C_DL, alpha=.75, label=nm)
        b = np.polyfit(d[xcol], d.g_pressured, 1); xs = np.linspace(d[xcol].min(), d[xcol].max(), 10); ax.plot(xs, np.polyval(b, xs), color="k", lw=1)
        ax.set_xlabel(lab, fontsize=9); ax.set_title(f"partial r = {tt.r:.2f} (95% CI {tt.lo:.2f} to {tt.hi:.2f}), n = {int(tt.n)}", fontsize=9); ax.grid(alpha=.3)
    axes[0].set_ylabel("pressure rate on pass-rush snaps (z within position)"); axes[0].legend(fontsize=8)
    fig.suptitle("Figure 5. DL: the second yard after the snap predicts pressure better than the first (weight and draft slot held fixed)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig5_dl_outcome.png", dpi=160); plt.close(fig)


def fig6_ol_within(g, base):
    ol = g[g.side == "OL"].copy()
    pm = player_means(ol, ["nfl_id"], ["depth_10", "set_angle_10", "t_d1_d2"], 100).set_index("nfl_id")
    x = ol[ol.nfl_id.isin(pm.index) & ol.lined_up_position.isin(["T", "G", "C"])].copy()
    x = x.merge(base[["nfl_id", "pick", "combine_weight"]], on="nfl_id").dropna(subset=["peak_pressure_probability_allowed", "dropback_duration", "pass_rushers_encountered"])
    x = x.dropna(subset=["t_d1_d2"])  # one model sample for both models
    for m in ["depth_10", "set_angle_10", "t_d1_d2"]:
        x["P_" + m] = x.nfl_id.map(pm[m]); x["w_" + m] = x[m] - x.groupby("nfl_id")[m].transform("mean")  # true within-player deviation
    f_between = "peak_pressure_probability_allowed ~ C(lined_up_position) + dropback_duration + pass_rushers_encountered + pick + combine_weight + P_depth_10 + P_set_angle_10 + P_t_d1_d2"
    mb = smf.ols(f_between, data=x).fit(cov_type="cluster", cov_kwds={"groups": x.nfl_id})
    xx = x
    mw = smf.ols("peak_pressure_probability_allowed ~ w_depth_10 + w_set_angle_10 + w_t_d1_d2 + dropback_duration + pass_rushers_encountered", data=xx).fit(cov_type="cluster", cov_kwds={"groups": xx.nfl_id})
    RES["ol_between_model"] = {k: {"coef": round(mb.params[k], 4), "se": round(mb.bse[k], 4), "p": round(mb.pvalues[k], 4)} for k in ["P_depth_10", "P_set_angle_10", "P_t_d1_d2"]}
    RES["ol_within_model"] = {k: {"coef": round(mw.params[k], 4), "se": round(mw.bse[k], 4), "p": round(mw.pvalues[k], 4)} for k in ["w_depth_10", "w_set_angle_10", "w_t_d1_d2"]}
    RES["ol_model_n"] = int(mb.nobs)
    # binned plot: deviation of set depth from the player's own mean vs pressure prob (residual on situation)
    x["resid"] = x.peak_pressure_probability_allowed - smf.ols("peak_pressure_probability_allowed ~ C(lined_up_position) + dropback_duration + pass_rushers_encountered", data=x).fit().fittedvalues
    x["bin"] = pd.qcut(x.w_depth_10, 10, labels=False)
    b = x.groupby("bin").agg(dev=("w_depth_10", "mean"), y=("resid", "mean"), se=("resid", lambda v: v.std() / np.sqrt(len(v))))
    pb = pm.join(x.groupby("nfl_id").resid.mean().rename("y"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(pb.depth_10, pb.y, color=C_OL, alpha=.75, s=40); axes[0].set_xlabel("player's mean set depth at 1.0 s (yd)"); axes[0].set_ylabel("pressure prob. allowed, net of position & situation")
    axes[0].set_title(f"Between players: no relation (coef p = {mb.pvalues['P_depth_10']:.2f}, n = {len(pb)})", fontsize=9); axes[0].grid(alpha=.3)
    axes[1].errorbar(b.dev, b.y, yerr=1.96 * b.se, fmt="o-", color=C_OL, capsize=3); axes[1].axhline(0, color="grey", lw=1)
    axes[1].set_xlabel("this snap's set depth minus the player's own mean (yd)"); axes[1].set_ylabel("")
    axes[1].set_title(f"Within a player: deeper-than-usual sets are the pressured ones\n(+{mw.params['w_depth_10']:.3f} per yd, n = {int(mw.nobs):,} snaps)", fontsize=9); axes[1].grid(alpha=.3)
    fig.suptitle("Figure 6. OL: set depth is a style, not a grade", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig6_ol_within.png", dpi=160); plt.close(fig)


def dl_play_level_models(g, base):
    """DL: pressure on a pass-rush snap vs the player's mean get-off / second
    yard, with alignment, blitz, down, distance, draft slot and weight."""
    dl = g[g.side == "DL"].copy()
    pm = player_means(dl, ["nfl_id"], ["t_d1_d2", "s_at_d1", "player_get_off"], 100).set_index("nfl_id")
    x = dl[dl.nfl_id.isin(pm.index)].merge(base[["nfl_id", "pick", "combine_weight", "ten_yd_split"]], on="nfl_id").dropna(subset=["down", "yards_to_go"])
    for m in ["t_d1_d2", "s_at_d1", "player_get_off"]:
        x["P_" + m] = x.nfl_id.map(pm[m])
    x["edge"] = x.lined_up_position.isin(["DE", "OLB"]).astype(float); x["down"] = x.down.astype(int)
    x["blitz"] = x.blitzing.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).fillna(0.0)
    out = {}
    for name, f in [("get_off", "pressured ~ edge + blitz + C(down) + yards_to_go + pick + combine_weight + P_player_get_off"),
                    ("second_yard", "pressured ~ edge + blitz + C(down) + yards_to_go + pick + combine_weight + P_t_d1_d2"),
                    ("both", "pressured ~ edge + blitz + C(down) + yards_to_go + pick + combine_weight + P_player_get_off + P_t_d1_d2"),
                    ("official_10yd", "pressured ~ edge + blitz + C(down) + yards_to_go + pick + combine_weight + ten_yd_split")]:
        xx = x.dropna(subset=["ten_yd_split"]) if "ten_yd_split" in f else x
        m = smf.ols(f, data=xx).fit(cov_type="cluster", cov_kwds={"groups": xx.nfl_id})
        out[name] = {"n": int(m.nobs), **{k: {"coef": round(m.params[k], 4), "se": round(m.bse[k], 4), "p": round(m.pvalues[k], 4)}
                                        for k in m.params.index if k.startswith("P_") or k == "ten_yd_split"}}
    RES["dl_play_level_models"] = out


def fig7_weekly(g, base):
    dl = g[g.side == "DL"].copy()
    wk = dl.groupby(["nfl_id", "season", "week"]).agg(v=("t_d1_d2", "mean"), n=("t_d1_d2", "size")).reset_index()
    wk = wk[wk.n >= 10]
    pm = player_means(dl, ["nfl_id"], ["t_d1_d2"], 100).set_index("nfl_id")
    game_sd = wk.groupby("nfl_id").v.std().median(); between_sd = pm.t_d1_d2.std()
    RES["weekly"] = {"per_game_sd": round(game_sd, 4), "between_player_sd": round(between_sd, 4), "games_for_r70": int(np.ceil(2.33 * (game_sd ** 2) / (between_sd ** 2)))}
    # pick 3 players with most games, show rolling 4-game mean
    top = wk.groupby("nfl_id").size().nlargest(3).index
    fig, ax = plt.subplots(figsize=(9, 4))
    names = base.set_index("nfl_id").display_name
    for i, pid in enumerate(top):
        s = wk[wk.nfl_id == pid].sort_values(["season", "week"]).reset_index(drop=True)
        xs = np.arange(len(s)); ax.plot(xs, s.v, "o", alpha=.35, color=f"C{i}", ms=4)
        ax.plot(xs, s.v.rolling(4, min_periods=2).mean(), color=f"C{i}", lw=2, label=f"{names[pid]} ({s.season.min()}–{s.season.max()})")
    ax.axhspan(pm.t_d1_d2.mean() - between_sd, pm.t_d1_d2.mean() + between_sd, color="grey", alpha=.12, label="±1 SD between players")
    ax.set_xlabel("game number (regular season, ≥10 rush snaps)"); ax.set_ylabel("time from 1 yd to 2 yd (s), lower = faster")
    ax.set_title(f"Figure 7. Week-to-week monitor of the second yard: one game is as noisy as the between-player spread\n(game SD {game_sd:.3f} s vs between-player SD {between_sd:.3f} s), so the line is a 4-game rolling mean", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "fig7_weekly.png", dpi=160); plt.close(fig)


def main():
    g, att, players, base = load_all()
    dol = player_table(g, att, base, "OL", "PASS_RUSH_DROP", ["pressure_allowed", "peak_pressure_probability_allowed"])
    ddl = player_table(g, att, base, "DL", "PASS_RUSH_DRILL", ["player_get_off", "pressured", "quick_pressure"])
    RES["n"] = {"ol_players": int(len(dol)), "dl_players": int(len(ddl)), "ol_snaps": int((g.side == "OL").sum()), "dl_snaps": int((g.side == "DL").sum()),
                "combine_ol_attempts": int((att.drill_name == "PASS_RUSH_DROP").sum()), "combine_dl_attempts": int((att.drill_name == "PASS_RUSH_DRILL").sum())}
    # official Combine numbers -> in-game (for the text)
    RES["official"] = {}
    for nm, d, ys in [("OL", dol, ["g_s_at_d1", "g_depth_10"]), ("DL", ddl, ["g_t_d1", "g_player_get_off", "g_pressured"])]:
        t = corr_table(d, ["forty", "ten_yd_split", "pick"], ys, controls=CTRL, n_boot=1000, seed=4)
        RES["official"][nm] = t.round(3).to_dict("records")
    RES["combine_onset"] = att[att.drill_name.isin(["PASS_RUSH_DROP", "PASS_RUSH_DRILL", "FORTY_YARD_DASH"])].groupby("drill_name").agg(
        attempts=("event_id", "size"), frac_moving_first_frame=("first_recorded_speed", lambda v: float((v >= 0.5).mean())),
        median_t_shift=("t_shift", "median")).round(3).to_dict("index")
    fig1_profiles(); fig2_reliability(g, att); fig3_transfer(dol, ddl); fig4_ol_scatter(dol); fig5_dl_outcome(ddl); fig6_ol_within(g, base); dl_play_level_models(g, base); fig7_weekly(g, base)
    (OUT.parent / "results.json").write_text(json.dumps(RES, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: RES[k] for k in ["n", "combine_onset", "ol_scatter", "dl_outcome", "weekly", "ol_between_model", "ol_within_model", "ol_model_n", "dl_play_level_models"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
