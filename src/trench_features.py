"""Shared kinematic features for trench (OL/DL) movement, used for BOTH
Combine drill attempts and in-game snaps so the two sides are measured
identically.

Input frame table columns (polars):
  key cols..., t (sec, 0 = movement start / ball snap), x, y, s, a,
  ux, uy  (unit axis vector: direction of "depth" = toward the offensive
           backfield for in-game, net displacement direction for a drill),
  o       (orientation deg, optional; NaN if absent),
  face    (target facing deg, optional).

NOTE: the Combine file's `dis` column is not a per-frame step (it behaves
like a squared step), so nothing here uses `dis`; distance comes from x/y.
"""
import polars as pl

WINDOWS = (0.5, 1.0, 1.5, 2.0)
TOL = 0.051  # half a frame at 10 Hz


def _at(col: str, tt: float, name: str | None = None) -> pl.Expr:
    """Value of `col` at the frame closest to time tt (within one frame)."""
    name = name or f"{col}_{int(round(tt * 10)):02d}"
    return pl.col(col).filter((pl.col("t") - tt).abs() < TOL).first().alias(name)


def _in(lo: float, hi: float) -> pl.Expr:
    return (pl.col("t") >= lo) & (pl.col("t") <= hi)


def add_depth_lateral(frames: pl.DataFrame, key: list[str], ref: str = "t0") -> pl.DataFrame:
    """Add depth (along axis) and signed lateral (perp. to axis) relative to
    the reference position of each group: the frame at t==0 (in-game, the
    snap) or the first recorded frame (Combine, ref="first"). If a
    `depth_shift` column exists (Combine onset back-extrapolation) it is
    added to depth."""
    if ref == "first":
        x0 = pl.col("x").sort_by("t").first().over(key); y0 = pl.col("y").sort_by("t").first().over(key)
    else:
        x0 = pl.col("x").filter(pl.col("t").abs() < TOL).first().over(key)
        y0 = pl.col("y").filter(pl.col("t").abs() < TOL).first().over(key)
    dx, dy = pl.col("x") - x0, pl.col("y") - y0
    shift = pl.col("depth_shift") if "depth_shift" in frames.columns else pl.lit(0.0)
    out = frames.with_columns(
        (dx * pl.col("ux") + dy * pl.col("uy") + shift).alias("depth"),
        (-dx * pl.col("uy") + dy * pl.col("ux")).alias("lat"),
    )
    if "o" in out.columns and "face" in out.columns:
        rel = ((pl.col("o") - pl.col("face") + 180.0) % 360.0 - 180.0).abs()
        out = out.with_columns(rel.alias("orel"))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("orel"))
    return out


def window_features(frames: pl.DataFrame, key: list[str]) -> pl.DataFrame:
    """One row per group with fixed-window kinematic features."""
    aggs = []
    for tt in WINDOWS:
        aggs += [_at("depth", tt), _at("lat", tt)]
    aggs += [
        _at("s", 0.5), _at("s", 1.0), _at("s", 1.5),
        pl.col("s").filter(_in(0.0, 1.5)).max().alias("smax_15"),
        pl.col("s").filter(_in(0.0, 1.0)).max().alias("smax_10"),
        pl.col("a").filter(_in(0.0, 0.5)).max().alias("amax_05"),
        pl.col("a").filter(_in(0.0, 1.0)).max().alias("amax_10"),
        pl.col("t").filter((pl.col("t") >= 0) & (pl.col("s") >= 1.0)).min().alias("t_s1"),
        pl.col("t").filter((pl.col("t") >= 0) & (pl.col("s") >= 2.0)).min().alias("t_s2"),
        pl.col("t").filter((pl.col("t") >= 0) & (pl.col("s") >= 3.0)).min().alias("t_s3"),
        pl.col("t").filter((pl.col("t") >= 0) & (pl.col("depth") >= 1.0)).min().alias("t_d1"),
        pl.col("t").filter((pl.col("t") >= 0) & (pl.col("depth") >= 2.0)).min().alias("t_d2"),
        pl.col("t").filter((pl.col("t") >= 0) & (pl.col("depth") >= 3.0)).min().alias("t_d3"),
        # onset-free quantities: speed when depth first reaches 1 / 2 yd
        pl.col("s").filter((pl.col("t") >= 0) & (pl.col("depth") >= 1.0)).first().alias("s_at_d1"),
        pl.col("s").filter((pl.col("t") >= 0) & (pl.col("depth") >= 2.0)).first().alias("s_at_d2"),
        _at("orel", 0.0, "orel_00"), _at("orel", 1.0, "orel_10"), _at("orel", 0.5, "orel_05"),
        _at("s", -0.2, "pre_s"),
        pl.len().alias("n_frames"),
        pl.col("t").min().alias("t_min"), pl.col("t").max().alias("t_max"),
    ]
    out = frames.group_by(key, maintain_order=True).agg(aggs)
    # set angle: 0 = straight back/forward along axis, 90 = pure lateral
    return out.with_columns(
        (pl.arctan2(pl.col("lat_10").abs(), pl.col("depth_10")).degrees()).alias("set_angle_10"),
        pl.col("lat_10").abs().alias("lat_abs_10"),
        pl.col("lat_15").abs().alias("lat_abs_15"),
        (pl.col("t_d2") - pl.col("t_d1")).alias("t_d1_d2"),
        # side of the set: lat < 0 is the offensive left in both files (an OL's own
        # left; a DL's own right)
        pl.when(pl.col("lat_10") < 0).then(pl.lit("L")).otherwise(pl.lit("R")).alias("side_10"),
    )


def speed_profiles(frames: pl.DataFrame, key: list[str], t_hi: float = 2.0) -> pl.DataFrame:
    """Per group: list columns of t, s, depth, lat, orel for 0 <= t <= t_hi
    (10 Hz), so curves can be compared later without re-reading tracking."""
    f = frames.filter((pl.col("t") >= -TOL) & (pl.col("t") <= t_hi + TOL)).sort(key + ["t"])
    return f.group_by(key, maintain_order=True).agg(
        pl.col("t").round(1).alias("t_list"), pl.col("s").alias("s_list"),
        pl.col("depth").alias("depth_list"), pl.col("lat").alias("lat_list"),
        pl.col("orel").alias("orel_list"),
    )


def extract(frames: pl.DataFrame, key: list[str]) -> pl.DataFrame:
    return window_features(add_depth_lateral(frames, key), key)
