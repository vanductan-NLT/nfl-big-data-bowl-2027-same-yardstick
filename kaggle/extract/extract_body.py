
# ---------------------------------------------------------------------------
# In-game extraction body (appended after trench_features.py by build script)
# ---------------------------------------------------------------------------
import glob
import os
import time as _time
import numpy as np

# competition files may sit one folder deeper than the slug; locate games.csv
BASE = os.path.dirname(glob.glob("/kaggle/input/**/games.csv", recursive=True)[0]) + "/"
print("BASE", BASE)
OL = ["T", "G", "C"]
DL = ["DT", "DE", "OLB", "NT"]
T_LO, T_HI = -0.3, 3.0
NA = ["NA", ""]  # competition CSVs encode nulls as NA

t0 = _time.time()
games = pl.read_csv(BASE + "games.csv", null_values=NA).filter(pl.col("season_type") == "REG")
reg_ids = games["game_id"].to_list()
players = pl.read_csv(BASE + "players.csv", null_values=NA).select("nfl_id", "nfl_position")
trench = players.filter(pl.col("nfl_position").is_in(OL + DL))
trench_ids = trench["nfl_id"].to_list()
print("REG games", len(reg_ids), "trench players", len(trench_ids))

PP_COLS = [
    "game_id", "play_id", "nfl_id", "lined_up_position", "play_direction", "quarter", "down",
    "yards_to_go", "offense_formation", "pass_result", "pass_length", "play_nullified_by_penalty",
    "yards_gained", "expected_points_added", "dropback_duration", "extended_sack_allowed",
    "pass_rushers_encountered", "peak_pressure_probability_allowed", "pressure_allowed",
    "sack_allowed", "time_to_pressure_allowed", "blitzing", "player_get_off", "sack",
    "time_to_pressure", "time_to_qb_hurry", "quick_pressure", "unblocked_pressure",
    "in_motion_at_ball_snap", "team_abbr", "possession_team",
]
pp = pl.read_csv(BASE + "player_play.csv", columns=PP_COLS, infer_schema_length=200000, null_values=NA)
pp = pp.filter(pl.col("nfl_id").is_in(trench_ids) & pl.col("game_id").is_in(reg_ids))
pp = pp.join(trench, on="nfl_id").join(games.select("game_id", "season", "week"), on="game_id")
pp = pp.with_columns(
    pl.when(pl.col("nfl_position").is_in(OL)).then(pl.lit("OL")).otherwise(pl.lit("DL")).alias("side")
)
# pass-pro snaps for OL, pass-rush snaps for DL
pp = pp.filter(
    ((pl.col("side") == "OL") & pl.col("dropback_duration").is_not_null())
    | ((pl.col("side") == "DL") & pl.col("player_get_off").is_not_null())
)
print("player_play trench pass rows", pp.shape, pp.group_by("side").len())

# facing target: OL faces the defense (+x when play goes right); DL faces offense
pp = pp.with_columns(
    pl.when(pl.col("play_direction") == "right").then(-1.0).otherwise(1.0).alias("ux"),
    pl.lit(0.0).alias("uy"),
    pl.when(pl.col("play_direction") == "right")
    .then(pl.when(pl.col("side") == "OL").then(90.0).otherwise(270.0))
    .otherwise(pl.when(pl.col("side") == "OL").then(270.0).otherwise(90.0))
    .alias("face"),
)
key = ["game_id", "play_id", "nfl_id"]
pp_axis = pp.select(key + ["ux", "uy", "face"])

parts = []
profiles = []
diag = []
for yr in (2023, 2024, 2025):
    t1 = _time.time()
    lf = pl.scan_csv(
        BASE + f"game_tracking_{yr}.csv",
        schema_overrides={"event": pl.String, "time": pl.String}, null_values=NA,
    )
    lf = lf.filter(pl.col("nfl_id").is_in(trench_ids) & pl.col("game_id").is_in(reg_ids))
    lf = lf.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f").alias("ts"))
    snap = (
        lf.filter(pl.col("event") == "ball_snap")
        .group_by("game_id", "play_id")
        .agg(pl.col("ts").min().alias("t_snap"))
    )
    df = (
        lf.join(snap, on=["game_id", "play_id"], how="inner")
        .with_columns(((pl.col("ts") - pl.col("t_snap")).dt.total_milliseconds() / 1000.0).alias("t"))
        .filter((pl.col("t") >= T_LO) & (pl.col("t") <= T_HI))
        .select(key + ["t", "x", "y", "s", "a", "o", "dir", "event"])
        .collect()
    )
    n_plays_snap = df.select("game_id", "play_id").unique().height
    df = df.join(pp_axis, on=key, how="inner").sort(key + ["t"])
    df = add_depth_lateral(df, key)
    feats = window_features(df, key)
    profiles.append(speed_profiles(df, key))
    # dominant movement direction in first 1 s (deg, dir column) for set-type work
    first_dir = (
        df.filter((pl.col("t") > 0.3) & (pl.col("t") <= 1.0))
        .group_by(key).agg(pl.col("dir").median().alias("dir_med_03_10"))
    )
    feats = feats.join(first_dir, on=key, how="left")
    parts.append(feats)
    diag.append({"season": yr, "plays_with_snap": n_plays_snap,
                 "player_plays": feats.height, "sec": round(_time.time() - t1, 1)})
    print(diag[-1], flush=True)

feats = pl.concat(parts)
out = pp.drop(["ux", "uy", "face"]).join(feats, on=key, how="inner")
out.write_csv("/kaggle/working/trench_game_features.csv")
pl.concat(profiles).join(pp.select(key + ["side", "lined_up_position", "season", "week"]), on=key, how="inner").write_parquet("/kaggle/working/trench_game_profiles.parquet")
pl.DataFrame(diag).write_csv("/kaggle/working/extract_diagnostics.csv")
print("OUT", out.shape, "elapsed", round(_time.time() - t0, 1), "s")
print(out.group_by("side", "nfl_position").len().sort("side", "nfl_position"))
print("pre-snap speed quantiles", out["pre_s"].quantile(0.5), out["pre_s"].quantile(0.95))
dl = out.filter(pl.col("side") == "DL").drop_nulls(["t_d1", "player_get_off"])
if dl.height:
    r = np.corrcoef(dl["t_d1"].to_numpy(), dl["player_get_off"].to_numpy())[0, 1]
    print("DL t_d1 vs player_get_off play-level r =", round(float(r), 3), "n =", dl.height)
print(out.select(["depth_10", "lat_abs_10", "s_10", "smax_15", "t_s2", "t_d1", "orel_10", "set_angle_10"]).describe())
