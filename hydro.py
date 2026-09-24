"""
hydro.py -- reservoir hydropower for PyPSA-Ethiopia.

Why this exists
---------------
Treating hydro as a generator capped by hour-to-hour runoff (what model.py
did before) makes a 74 km3 reservoir behave like a river with no dam: it can
only generate when rain happens to fall that hour. Ethiopia is ~90% hydro, so
that single simplification understated national supply badly.

Here each large plant is a PyPSA StorageUnit:

    inflow     real ERA5 runoff SHAPE, scaled so an average year delivers the
               plant's DESIGN energy (the long-run water budget)
    storage    live volume x head, capped at one year of design energy
    spill      free, when the reservoir is full

The window problem, and the rule curve
--------------------------------------
Models here usually run a window (e.g. 14 days), not a full year. A cyclic
reservoir over a January window cannot draw down, yet January is exactly when
Ethiopian reservoirs release water stored during the Kiremt rains.

So for each plant a full-year RULE CURVE is simulated first: constant release
at the long-run average, storage bounded by the reservoir. Any window then
starts at the curve's level on its first day and must end at the curve's level
on its last day. Inside the window the optimiser is free to shape dispatch
hour by hour; across windows, the fleet can never release more than its
long-run average. That is the "sustainable" mode.

The "drawdown" mode (base-year validation) instead starts every reservoir full
and leaves the end free -- which is what happened in FY2025/26, when GERD ran
~2.5 TWh above its design energy on water stored during filling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
HYDRO_PATH = DATA_DIR / "hydro_plants.csv"

HOURS_PER_YEAR = 8760
RHO_G = 1000.0 * 9.81          # kg/m3 * m/s2
TURBINE_EFF = 0.90


def load_hydro_plants(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    path = path or HYDRO_PATH
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    for col in ("capacity_mw", "design_gwh", "live_km3", "head_m",
                "first_year", "full_year", "first_share"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["first_share"] = df["first_share"].fillna(1.0)
    return df


def capacity_in_year(plant: pd.Series, year: int) -> float:
    """Commissioning ramp: 0 before first_year, partial, then nameplate."""
    if pd.notna(plant.first_year) and year < plant.first_year:
        return 0.0
    if pd.notna(plant.full_year) and year < plant.full_year:
        return float(plant.capacity_mw * plant.first_share)
    return float(plant.capacity_mw)


def storage_mwh(plant: pd.Series) -> float:
    """
    Usable reservoir energy: live volume x head x g x efficiency, capped at one
    year of design energy (a one-year model cannot use more than that).
    """
    design_mwh = float(plant.design_gwh) * 1000.0
    if pd.isna(plant.live_km3) or pd.isna(plant.head_m):
        return design_mwh
    joules = float(plant.live_km3) * 1e9 * RHO_G * float(plant.head_m) * TURBINE_EFF
    physical_mwh = joules / 3.6e9
    return float(min(physical_mwh, design_mwh))


def annual_inflow_mw(
    plant: pd.Series,
    year: int,
    shape_fn: Callable[[pd.DatetimeIndex, str], pd.Series],
    inflow_factor: float = 1.0,
) -> pd.Series:
    """
    Hourly inflow for a full calendar year, in MW of electric equivalent.

    The shape is the bus's runoff profile; the level is set so the year's total
    equals design energy x inflow_factor (1.0 = average year, 0.8 = the kind of
    20% shortfall EEP reported in September 2026).
    """
    idx = pd.date_range(f"{year}-01-01", periods=HOURS_PER_YEAR, freq="h")
    shape = shape_fn(idx, plant.bus).reindex(idx).fillna(0.0).clip(lower=0.0)
    mean = float(shape.mean())
    if mean <= 0:
        shape = pd.Series(1.0, index=idx)
        mean = 1.0
    mean_mw = float(plant.design_gwh) * 1000.0 / HOURS_PER_YEAR * inflow_factor
    return shape / mean * mean_mw


def rule_curve(
    inflow_mw: pd.Series, p_nom: float, e_max: float, passes: int = 3
) -> pd.Series:
    """
    Reservoir level (MWh, end of each hour) under a constant-release policy.

    Release is the long-run mean inflow, limited by turbine capacity. The level
    is bounded by [0, e_max]: excess spills, and when empty the release falls
    short. The year is simulated repeatedly until the end level feeds back into
    the start, giving a cyclic steady state.
    """
    inflow = inflow_mw.to_numpy()
    release = min(float(inflow.mean()), p_nom)
    level = 0.5 * e_max
    out = np.empty_like(inflow)
    for _ in range(passes):
        for t, q in enumerate(inflow):
            level = min(max(level + q - release, 0.0), e_max)
            out[t] = level
    return pd.Series(out, index=inflow_mw.index)


def window_inflow(inflow_hourly: pd.Series, sn: pd.DatetimeIndex, res_h: int) -> pd.Series:
    """Average the hourly inflow onto the model's snapshots."""
    key = pd.MultiIndex.from_arrays(
        [inflow_hourly.index.month, inflow_hourly.index.day, inflow_hourly.index.hour]
    )
    lookup = pd.Series(inflow_hourly.to_numpy(), index=key)
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    vals = []
    for t in sn:
        hours = pd.date_range(t, periods=res_h, freq="h")
        k = pd.MultiIndex.from_arrays([hours.month, hours.day, hours.hour])
        vals.append(float(lookup.reindex(k).ffill().bfill().mean()))
    return pd.Series(vals, index=sn)


def curve_level_at(curve: pd.Series, when: pd.Timestamp) -> float:
    """Rule-curve level at a (month, day, hour), matched across years."""
    mask = ((curve.index.month == when.month) & (curve.index.day == when.day)
            & (curve.index.hour == when.hour))
    hits = curve[mask]
    if len(hits):
        return float(hits.iloc[0])
    return float(curve.iloc[-1])


def calibrate_to_cf(shape: pd.Series, target_cf: float, iters: int = 40) -> pd.Series:
    """
    Scale a 0..1 shape so that, after clipping at 1, its mean hits target_cf.

    Used for existing wind farms (ERA5 regional shape, observed level) and for
    run-of-river hydro (runoff shape, design CF). Bisection on the scale.
    """
    s = shape.clip(lower=0.0).fillna(0.0)
    if s.mean() <= 0 or target_cf <= 0:
        return pd.Series(0.0, index=shape.index)
    lo, hi = 0.0, 1.0 / max(float(s[s > 0].min()), 1e-9)
    hi = max(hi, target_cf / float(s.mean()) * 4)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float((s * mid).clip(upper=1.0).mean()) < target_cf:
            lo = mid
        else:
            hi = mid
    return (s * hi).clip(upper=1.0)
