"""
timeseries.py -- make profile functions resolution-safe.

A snapshot at 12:00 with 12-hour resolution stands for the whole 12:00-24:00
block. Evaluating a profile only AT 12:00 is point sampling: at 12 h
resolution it sees midnight and noon and never the 19:00 evening peak, so
demand came out ~9% low, and it sees solar only at noon, which roughly doubles
solar output.

@interval_mean fixes this at the source. It expands the snapshot index to
hourly, calls the wrapped function on the hourly index, and averages each
block back onto its snapshot. At 1-hour resolution it is a no-op.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd


def _step_hours(idx: pd.DatetimeIndex) -> int:
    if len(idx) < 2:
        return 1
    step = (idx[1] - idx[0]) / pd.Timedelta(hours=1)
    return max(int(round(step)), 1)


def expand_hourly(sn: pd.DatetimeIndex, res_h: int) -> pd.DatetimeIndex:
    offsets = pd.to_timedelta(np.tile(np.arange(res_h), len(sn)), unit="h")
    return pd.DatetimeIndex(sn.repeat(res_h) + offsets)


def interval_mean(fn):
    """Decorator: the first DatetimeIndex argument is treated as snapshots."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        pos = next((i for i, a in enumerate(args) if isinstance(a, pd.DatetimeIndex)), None)
        if pos is None:
            return fn(*args, **kwargs)
        sn = args[pos]
        res = _step_hours(sn)
        if res <= 1:
            return fn(*args, **kwargs)
        hourly = expand_hourly(sn, res)
        new_args = list(args)
        new_args[pos] = hourly
        out = fn(*new_args, **kwargs)
        if not isinstance(out, pd.Series) or len(out) != len(hourly):
            return out
        vals = out.to_numpy(dtype=float).reshape(len(sn), res).mean(axis=1)
        return pd.Series(vals, index=sn, name=out.name)
    return wrapper
