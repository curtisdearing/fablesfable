"""Fit the Phase 6.4 weather constants from history -- replacing factors.py's
guessed severity (0.55·wind/30 + 0.30·precip/8mm + 0.15·cold<20F).

Three fits, 2019-2023 REG only (2024+ held out for checkpoint evals):

  PASSING   team-game pass yards ~ trailing team pass yards + wind + cold +
            precip + crosswind share (outdoor-effective games only).
  FG        make probability ~ distance + wind + cold + precip + along/cross
            + Denver (logistic, all FG attempts).
  ROOF      retractable-roof behavior: P(open) vs temp/precip -- answers
            whether an open roof ever coincides with weather worth pricing.

Run:  python3 scripts/fit_weather.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue.weather_study import build_game_weather  # noqa: E402

FIT_SEASONS = (2019, 2023)


def _ols(X: np.ndarray, y: np.ndarray, labels):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = np.sum(resid ** 2) / dof * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    for l, b, s in zip(labels, beta, se):
        print(f"    {l:>18}: {b:+9.3f}  (t={b/s:+5.1f})")
    return beta, se


def _logit(X: np.ndarray, y: np.ndarray, labels, iters: int = 60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):                      # Newton-Raphson
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        b = b + np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ b))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]))))
    for l, bb, s in zip(labels, b, se):
        print(f"    {l:>18}: {bb:+9.4f}  (t={bb/s:+5.1f})")
    return b, se


def main() -> None:
    from nflvalue.features import _team_week, load_pbp
    gw = build_game_weather()
    gw = gw[gw["season"].between(*FIT_SEASONS)]
    print(f"[truth] {len(gw):,} games; outdoor-effective {gw['effective_outdoor'].mean():.0%}, "
          f"precip flag {gw['precip_flag'].mean():.0%}, wind dir parsed "
          f"{gw['wind_dir_deg'].notna().mean():.0%}")

    # ---------------- passing ---------------- #
    pbp = load_pbp()
    tw = _team_week(pbp)
    tw = tw.sort_values(["team", "season", "week"]).reset_index(drop=True)
    pyd = (pbp[pbp["pass_attempt"] == 1]
           .groupby(["season", "week", "posteam", "game_id"])["passing_yards"]
           .apply(lambda s: np.nansum(s.to_numpy())).rename("pass_yds").reset_index()
           .rename(columns={"posteam": "team"}))
    pyd["trail"] = (pyd.sort_values(["team", "season", "week"])
                    .groupby("team")["pass_yds"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean()))
    d = pyd.merge(gw, on="game_id", how="inner").dropna(subset=["trail"])
    d = d[d["season_x"].between(*FIT_SEASONS)] if "season_x" in d else d
    out = d[d["effective_outdoor"] & d["wind_mph"].notna() & d["temp_f"].notna()].copy()
    out["wind10"] = np.maximum(out["wind_mph"] - 10, 0)      # yards lost per mph ABOVE 10
    out["cold"] = np.maximum(32 - out["temp_f"], 0)          # degrees below freezing
    out["cross_share"] = (out["cross_mph"] / out["wind_mph"].clip(lower=1)).fillna(0.5)
    X = np.column_stack([np.ones(len(out)), out["trail"], out["wind_mph"],
                         out["wind10"], out["cold"], out["precip_flag"],
                         out["wind_mph"] * out["cross_share"]])
    print(f"\n[passing] n={len(out):,} outdoor team-games, pass_yds OLS:")
    _ols(X, out["pass_yds"].to_numpy(),
         ["const", "trail_pass_yds", "wind_mph(0-10 incl)", "wind_mph>10",
          "deg_below_32F", "precip_flag", "crosswind_mph"])

    # ---------------- field goals ---------------- #
    import glob
    cols = ["game_id", "season", "season_type", "field_goal_attempt",
            "field_goal_result", "kick_distance"]
    frames = [pd.read_parquet(os.path.join(ROOT, "historical", "historical_pbp.parquet"), columns=cols)]
    for fn in sorted(glob.glob(os.path.join(ROOT, "historical", "pbp_*.parquet"))):
        frames.append(pd.read_parquet(fn, columns=cols))
    fg = pd.concat(frames, ignore_index=True)
    fg = fg[(fg["season_type"] == "REG") & (fg["field_goal_attempt"] == 1)
            & fg["kick_distance"].notna() & fg["season"].between(*FIT_SEASONS)]
    fg = fg.merge(gw, on="game_id", how="inner")
    fg["made"] = (fg["field_goal_result"] == "made").astype(float)
    f = fg[fg["wind_mph"].notna() & fg["temp_f"].notna()].copy()
    f["wind_eff"] = np.where(f["effective_outdoor"], f["wind_mph"], 0.0)
    f["along_eff"] = np.where(f["effective_outdoor"], f["along_mph"].fillna(f["wind_mph"] * 0.64), 0.0)
    f["cross_eff"] = np.where(f["effective_outdoor"], f["cross_mph"].fillna(f["wind_mph"] * 0.64), 0.0)
    f["cold"] = np.where(f["effective_outdoor"], np.maximum(32 - f["temp_f"], 0), 0.0)
    f["dist50"] = np.maximum(f["kick_distance"] - 50, 0)
    Xf = np.column_stack([np.ones(len(f)), f["kick_distance"], f["dist50"],
                          f["along_eff"], f["cross_eff"], f["cold"],
                          f["precip_flag"] * f["effective_outdoor"], f["is_denver"]])
    print(f"\n[FG] n={len(f):,} attempts, logistic P(make):")
    _logit(Xf, f["made"].to_numpy(),
           ["const", "distance_yds", "distance>50", "alongwind_mph",
            "crosswind_mph", "deg_below_32F", "precip_flag", "denver"])
    # headline effect sizes at 45 yds
    print("    (interpretation printed by decisions_p6.md)")

    # ---------------- retractable roofs ---------------- #
    gw_all = build_game_weather()
    RETRACT = {"ARI", "ATL", "DAL", "HOU", "IND"}
    r = gw_all[gw_all["home_team"].isin(RETRACT)].copy()
    r["is_open"] = (r["roof"] == "open").astype(int)
    print(f"\n[roof] retractable games n={len(r):,}, open {r['is_open'].mean():.1%}")
    op = r[r["is_open"] == 1]
    print(f"    open-roof games: temp {op['temp_f'].min():.0f}-{op['temp_f'].max():.0f}F "
          f"(mean {op['temp_f'].mean():.0f}), precip {op['precip_flag'].mean():.0%}, "
          f"wind mean {op['wind_mph'].mean():.1f} mph")
    cl = r[r["is_open"] == 0]
    print(f"    closed-roof games: temp mean {cl['temp_f'].mean():.0f}F, "
          f"precip {cl['precip_flag'].mean():.0%}")
    bad_open = op[(op["temp_f"] < 40) | (op["precip_flag"] == 1) | (op["wind_mph"] >= 15)]
    print(f"    open WITH pricable weather (cold<40F/precip/wind>=15): {len(bad_open)} games")


if __name__ == "__main__":
    main()
