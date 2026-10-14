"""Build per-attempt Combine kinematic features for trench drills.

Usage: python src/build_combine_trench_metrics.py
Writes data/processed/combine_trench_attempts.csv
"""
from pathlib import Path
import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trench_features import add_depth_lateral, speed_profiles, window_features  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# drill name normalisation (2023 vs 2024-25 spellings)
DRILL_ALIAS = {"PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL", "LINE": "LINE_DRILL"}
DRILLS = [
    "PASS_RUSH_DROP", "PASS_PRO_MIRROR_DRILL", "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE",
    "PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL", "FORTY_YARD_DASH",
]
START_SPEED = 0.5  # yd/s: movement onset defines t = 0 for a drill attempt
# The Combine file has no start signal and most attempts are already moving
# on their first recorded frame. Onset is back-extrapolated assuming constant
# acceleration from rest over the first frames: t_shift = v0/a0, d_shift =
# v0^2/(2 a0). Attempts whose first frame is already fast are excluded.
A0_FLOOR = 1.5      # yd/s^2 floor for the extrapolation acceleration
MAX_FIRST_SPEED = 2.0  # yd/s: first recorded frame faster than this -> drop attempt


def hitch_features(df: pl.DataFrame) -> pl.DataFrame:
    """OL PASS_RUSH_DROP has a kick-slide phase, a hitch (speed dip), then a
    sprint. Find first speed peak (>1 yd/s), the following dip, and the time
    from dip to 5 yd/s (redirect burst)."""
    rows = []
    for (ev,), g in df.group_by("event_id", maintain_order=True):
        t = g["t"].to_numpy(); s = g["s"].to_numpy(); d = g["depth"].to_numpy()
        m = t >= 0
        t, s, d = t[m], s[m], d[m]
        rec = {"event_id": ev, "peak1_t": np.nan, "peak1_s": np.nan, "dip_t": np.nan,
               "dip_s": np.nan, "dip_depth": np.nan, "redirect_t": np.nan}
        if len(s) < 8:
            rows.append(rec); continue
        # smooth 3 frames
        ss = np.convolve(s, np.ones(3) / 3, mode="same")
        peaks = [i for i in range(1, len(ss) - 1) if ss[i] >= ss[i - 1] and ss[i] > ss[i + 1] and ss[i] > 1.0]
        if not peaks:
            rows.append(rec); continue
        p = peaks[0]
        rec["peak1_t"], rec["peak1_s"] = t[p], ss[p]
        dips = [i for i in range(p + 1, len(ss) - 1) if ss[i] <= ss[i - 1] and ss[i] < ss[i + 1]]
        if dips:
            q = dips[0]
            rec["dip_t"], rec["dip_s"], rec["dip_depth"] = t[q], ss[q], d[q]
            after = np.where((np.arange(len(ss)) > q) & (ss >= 5.0))[0]
            if len(after):
                rec["redirect_t"] = t[after[0]] - t[q]
        rows.append(rec)
    return pl.DataFrame(rows)


def main() -> None:
    ct = pl.read_csv(RAW / "combine_tracking.csv", schema_overrides={"time": pl.String})
    ct = ct.filter(pl.col("entity_type") == "PLAYER").with_columns(
        pl.col("drill_name").replace(DRILL_ALIAS).alias("drill_name")
    ).filter(pl.col("drill_name").is_in(DRILLS))
    ct = ct.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f").alias("ts"))
    ct = ct.sort(["event_id", "ts"])

    # movement onset per attempt: first frame with s >= START_SPEED, then
    # back-extrapolated to rest (see A0_FLOOR); frames before onset dropped
    key = ["event_id"]
    onset = pl.col("ts").filter(pl.col("s") >= START_SPEED).first().over(key)
    ct = ct.with_columns(((pl.col("ts") - onset).dt.total_milliseconds() / 1000.0).alias("t"),
                         pl.col("s").sort_by("ts").first().over(key).alias("first_recorded_speed"))
    ct = ct.filter(pl.col("t").is_not_null() & (pl.col("t") >= 0))
    v0 = pl.col("s").sort_by("t").first().over(key)
    a0 = pl.max_horizontal(pl.col("a").sort_by("t").head(3).mean().over(key), pl.lit(A0_FLOOR))
    ct = ct.with_columns(v0.alias("first_speed"), (v0 / a0).alias("t_shift"), (v0 ** 2 / (2 * a0)).alias("depth_shift"))
    ct = ct.filter(pl.col("first_speed") < MAX_FIRST_SPEED).with_columns((pl.col("t") + pl.col("t_shift")).alias("t"))

    # axis: the two primary drills are run along the field x axis from a stance
    # facing +x (OL kick-slides to -x, DL bursts to +x), so use that axis so
    # that depth/lat mean the same thing as in-game (lat < 0 = player's left).
    # Other drills: unit vector of net displacement from onset to last frame.
    first_x = pl.col("x").sort_by("t").first().over(key); first_y = pl.col("y").sort_by("t").first().over(key)
    last_x = pl.col("x").sort_by("t").last().over(key); last_y = pl.col("y").sort_by("t").last().over(key)
    ct = ct.with_columns((last_x - first_x).alias("_dx"), (last_y - first_y).alias("_dy"))
    norm = pl.max_horizontal((pl.col("_dx") ** 2 + pl.col("_dy") ** 2).sqrt(), pl.lit(1e-6))
    ct = ct.with_columns(
        pl.when(pl.col("drill_name") == "PASS_RUSH_DROP").then(-1.0)
        .when(pl.col("drill_name") == "PASS_RUSH_DRILL").then(1.0)
        .otherwise(pl.col("_dx") / norm).alias("ux"),
        pl.when(pl.col("drill_name").is_in(["PASS_RUSH_DROP", "PASS_RUSH_DRILL"])).then(0.0)
        .otherwise(pl.col("_dy") / norm).alias("uy"),
    )

    meta = ct.group_by(key, maintain_order=True).agg(
        pl.col("nfl_id").first(), pl.col("draft_year").first(), pl.col("drill_type").first(),
        pl.col("drill_name").first(), pl.col("attempt").first(),
        norm.first().alias("net_disp"), pl.col("s").max().alias("smax_all"),
        pl.col("t").max().alias("duration"), pl.col("first_speed").first(), pl.col("t_shift").first(),
        pl.col("first_recorded_speed").first(),
    )
    fr = add_depth_lateral(ct, key, ref="first")
    feats = window_features(fr, key)
    hitch = hitch_features(fr.filter(pl.col("drill_name") == "PASS_RUSH_DROP"))
    out = meta.join(feats, on=key).join(hitch, on=key, how="left")
    out.write_csv(OUT / "combine_trench_attempts.csv")
    speed_profiles(fr, key).join(meta, on=key).write_parquet(OUT / "combine_trench_profiles.parquet")
    print(out.shape)
    print(out.group_by("drill_name").agg(pl.len(), pl.col("nfl_id").n_unique(), pl.col("first_recorded_speed").median(),
                                         (pl.col("first_recorded_speed") >= START_SPEED).mean().alias("frac_moving_at_first_frame"),
                                         pl.col("t_shift").median()).sort("drill_name"))
    cols = ["depth_05", "depth_10", "depth_15", "s_05", "s_10", "smax_15", "amax_05", "t_s2", "t_d1", "set_angle_10", "peak1_s", "dip_t", "redirect_t"]
    for dn in ["PASS_RUSH_DROP", "PASS_RUSH_DRILL"]:
        print(dn)
        print(out.filter(pl.col("drill_name") == dn).select(cols).describe())


if __name__ == "__main__":
    main()
