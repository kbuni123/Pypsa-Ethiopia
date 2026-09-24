"""
model.py -- PyPSA-Ethiopia core model.

Deliberately free of Streamlit imports so it can be unit-tested headless:

    python -m model        # runs a self-test and prints KPIs

THE RULE THAT BROKE THE OLD APP
-------------------------------
`pypsa.Network()` starts with a single dummy snapshot ("now").  Any code that
does `np.random.uniform(..., len(n.snapshots))` *before* `n.set_snapshots(...)`
builds a length-0/length-1 array, which silently becomes zero demand and zero
generator availability.  The optimisation then returns EUR 0 / 0 MWh / 0%
renewables, which looks like a solver failure but is really an empty model.

`build_network()` below sets snapshots as step 1 and every profile function
takes the snapshot index as an explicit argument, so the failure mode cannot
recur.  `validate_network()` asserts it anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pypsa

import hydro as H
from timeseries import interval_mean

try:
    import data as D
    DATA_AVAILABLE = True
except Exception:  # data.py absent or its deps missing
    D = None
    DATA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Per-process cache for the profile CSVs so each run reads them once.
_PROFILE_CACHE: dict = {}

HOURS_PER_YEAR = 8760


# --------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Region:
    name: str
    lat: float
    lon: float
    load_share: float      # fraction of national demand
    clearness: float       # mean clear-sky transmittance -> solar quality
    wind_cf: float         # mean onshore wind capacity factor
    hydro_mw: float        # existing hydro capacity attributed to the node


REGIONS: List[Region] = [
    Region("Addis_Ababa", 9.03, 38.74, 0.38, 0.70, 0.18, 0.0),
    Region("Bahir_Dar",  11.59, 37.39, 0.13, 0.72, 0.22, 6450.0),   # Blue Nile / GERD area
    Region("Mekelle",    13.49, 39.47, 0.12, 0.80, 0.32, 0.0),      # Ashegoda wind
    Region("Hawassa",     7.05, 38.48, 0.20, 0.71, 0.20, 2100.0),   # Gibe cascade
    Region("Dire_Dawa",   9.60, 41.87, 0.12, 0.78, 0.28, 0.0),
    Region("Jigjiga",     9.35, 42.80, 0.05, 0.79, 0.30, 0.0),
]

# (bus0, bus1, km, existing MW)
CORRIDORS: List[Tuple[str, str, float, float]] = [
    ("Addis_Ababa", "Bahir_Dar", 480.0, 1200.0),
    ("Addis_Ababa", "Hawassa",   275.0, 1400.0),
    ("Addis_Ababa", "Dire_Dawa", 445.0,  900.0),
    ("Addis_Ababa", "Mekelle",   780.0,  700.0),
    ("Bahir_Dar",   "Mekelle",   540.0,  400.0),
    ("Dire_Dawa",   "Jigjiga",   200.0,  300.0),
]


# --------------------------------------------------------------------------
# Technology cost assumptions (overnight EUR/MW, real)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Tech:
    carrier: str
    capex: float           # EUR per MW
    fom_pct: float         # fixed O&M, % of capex per year
    lifetime: int          # years
    marginal_cost: float   # EUR per MWh
    co2: float = 0.0       # t CO2 per MWh electrical


TECHS: Dict[str, Tech] = {
    "solar":      Tech("solar",      550_000, 0.020, 25,   0.0),
    "wind":       Tech("wind",     1_250_000, 0.025, 25,   0.0),
    "hydro":      Tech("hydro",    2_300_000, 0.015, 60,   0.0),
    "geothermal": Tech("geothermal", 3_600_000, 0.030, 30,  4.0),
    "diesel":     Tech("diesel",     450_000, 0.030, 25, 185.0, co2=0.72),
    "battery":    Tech("battery",    620_000, 0.020, 15,   0.0),  # per MW incl. 4 h of energy
}

VOLL = 3000.0  # EUR/MWh -- value of lost load, used by the slack generator


def annuity(lifetime: int, discount_rate: float) -> float:
    """Capital recovery factor."""
    if discount_rate <= 0:
        return 1.0 / lifetime
    return discount_rate / (1.0 - (1.0 + discount_rate) ** (-lifetime))


def annual_fixed_cost(tech: Tech, discount_rate: float) -> float:
    """Annualised CAPEX + fixed O&M in EUR/MW/year."""
    return tech.capex * (annuity(tech.lifetime, discount_rate) + tech.fom_pct)


# --------------------------------------------------------------------------
# Scenario configuration
# --------------------------------------------------------------------------
@dataclass
class ScenarioConfig:
    start: str = "2030-01-01"
    days: int = 365          # reservoirs need a full year to shift water between seasons
    resolution_hours: int = 6
    peak_demand_mw: float = 6500.0       # national coincident peak
    discount_rate: float = 0.08
    enabled: Dict[str, bool] = field(default_factory=lambda: {
        "solar": True, "wind": True, "hydro": True,
        "geothermal": True, "diesel": True, "battery": True,
    })
    existing_hydro: bool = True
    co2_limit_t: float | None = None     # None = unconstrained
    expand_network: bool = True
    solver: str = "highs"
    # Real-data switches. Each falls back silently to synthetic if the
    # underlying file is missing; data.provenance() reports what was used.
    use_real_weather: bool = True
    use_real_plants: bool = True
    use_real_demand: bool = True
    use_real_costs: bool = True
    use_available_capacity: bool = True   # available_mw vs nameplate capacity_mw
    use_real_grid: bool = True
    grid_source: str = "auto"             # auto | osm | wb2007
    # Hydro: "sustainable" = reservoirs end the year where they start (future
    # years); "drawdown" = start full and may end lower (base-year validation,
    # reproduces GERD's post-filling drawdown).
    hydro_mode: str = "sustainable"
    inflow_factor: float = 1.0            # 0.8 ~ the 2026 El Nino shortfall
    mining_status: str = "continue"       # continue / curtail / terminate
    # EUR/MWh the optimiser pays to curtail miners. None = demand yaml value
    # (the 28.9 EUR/MWh tariff). Raise it to price in the foreign-currency
    # revenue and policy value EEP attaches to mining.
    mining_curtail_price: float | None = None
    underlying_growth: float | None = None  # None = value in demand yaml
    include_access: bool = True
    grid_access_target: float | None = None
    grid_access_target_year: int | None = None
    kwh_per_new_household: float | None = None
    allow_new_build: bool = True          # False = existing system only (validation)

    @property
    def year(self) -> int:
        return int(str(self.start)[:4])

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Deterministic profiles (same inputs -> same outputs, no np.random surprises)
# --------------------------------------------------------------------------
def make_snapshots(cfg: ScenarioConfig) -> pd.DatetimeIndex:
    periods = int(cfg.days * 24 / cfg.resolution_hours)
    return pd.date_range(
        start=cfg.start, periods=periods, freq=f"{cfg.resolution_hours}h"
    )


@interval_mean
def solar_profile(sn: pd.DatetimeIndex, region: Region) -> pd.Series:
    """Clear-sky geometry x regional clearness x Kiremt rain-season damping."""
    doy = sn.dayofyear.to_numpy()
    # local solar time, Ethiopia is UTC+3 and the index is treated as local
    hour = sn.hour.to_numpy() + sn.minute.to_numpy() / 60.0
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    omega = np.radians(15.0 * (hour - 12.0))
    lat = np.radians(region.lat)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(omega)
    cos_zenith = np.clip(cos_zenith, 0.0, 1.0)

    # June-September Kiremt rains cut irradiance, strongest in the highlands
    rain = 1.0 - 0.28 * np.exp(-((doy - 213) ** 2) / (2 * 38.0 ** 2))
    cf = cos_zenith * region.clearness * rain
    return pd.Series(np.clip(cf, 0.0, 1.0), index=sn)


@interval_mean
def wind_profile(sn: pd.DatetimeIndex, region: Region, seed: int = 2030) -> pd.Series:
    """Smooth AR(1) series scaled to the region's mean capacity factor."""
    rng = np.random.default_rng(seed + int(abs(region.lat * 100)))
    n = len(sn)
    noise = rng.standard_normal(n)
    x = np.zeros(n)
    phi = 0.88
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi ** 2) * noise[i]

    # diurnal component: Ethiopian highland wind peaks in the afternoon/evening
    hour = sn.hour.to_numpy()
    diurnal = 1.0 + 0.30 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    # Bega (Oct-Feb) is the windy season
    doy = sn.dayofyear.to_numpy()
    seasonal = 1.0 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365.0)

    raw = np.clip(0.5 + 0.22 * x, 0.0, 1.0) * diurnal * seasonal
    cf = raw * (region.wind_cf / max(raw.mean(), 1e-9))
    return pd.Series(np.clip(cf, 0.0, 1.0), index=sn)


@interval_mean
def hydro_profile(sn: pd.DatetimeIndex) -> pd.Series:
    """Seasonal inflow availability: wet Jul-Sep, dry Feb-May."""
    doy = sn.dayofyear.to_numpy()
    cf = 0.55 + 0.35 * np.sin(2 * np.pi * (doy - 150) / 365.0)
    return pd.Series(np.clip(cf, 0.15, 0.95), index=sn)


@interval_mean
def load_profile(sn: pd.DatetimeIndex, region: Region, peak_mw: float) -> pd.Series:
    """Evening-peaking demand with a weekend dip and a hot-season uplift."""
    hour = sn.hour.to_numpy()
    shape = (
        0.62
        + 0.10 * np.exp(-((hour - 11) ** 2) / (2 * 3.0 ** 2))   # midday commercial
        + 0.38 * np.exp(-((hour - 19) ** 2) / (2 * 2.2 ** 2))   # evening residential
    )
    weekend = np.where(sn.dayofweek.to_numpy() >= 5, 0.90, 1.0)
    doy = sn.dayofyear.to_numpy()
    seasonal = 1.0 + 0.06 * np.cos(2 * np.pi * (doy - 120) / 365.0)

    series = shape * weekend * seasonal
    series = series / series.max()                    # normalise to a 1.0 peak
    return pd.Series(series * peak_mw * region.load_share, index=sn)


# --------------------------------------------------------------------------
# Real data first, synthetic fallback
#
# Each of these tries data.py and drops back to the synthetic function above if
# the real data is not there. Nothing silently pretends to be real: what
# actually got used is reported by data.provenance() and shown in the app.
# --------------------------------------------------------------------------
def _real_profile(kind: str, sn: pd.DatetimeIndex, bus: str):
    if not DATA_AVAILABLE:
        return None
    try:
        return D.profile_for(kind, sn, bus, cache=_PROFILE_CACHE)
    except Exception as exc:
        logger.warning("real %s profile unavailable for %s: %s", kind, bus, exc)
        return None


def solar_cf(sn: pd.DatetimeIndex, region: Region) -> pd.Series:
    real = _real_profile("solar", sn, region.name)
    return real if real is not None else solar_profile(sn, region)


def wind_cf(sn: pd.DatetimeIndex, region: Region) -> pd.Series:
    real = _real_profile("wind", sn, region.name)
    return real if real is not None else wind_profile(sn, region)


def hydro_cf(sn: pd.DatetimeIndex, region: Region) -> pd.Series:
    real = _real_profile("hydro", sn, region.name)
    return real if real is not None else hydro_profile(sn)


def demand_series(sn: pd.DatetimeIndex, region: Region, cfg) -> pd.Series:
    """Anchored to published annual energy when data/ is populated."""
    if DATA_AVAILABLE and cfg.use_real_demand:
        anchors = D.load_demand_anchors()
        if anchors is not None:
            return D.anchored_load_profile(sn, region.name, anchors)
    return load_profile(sn, region, cfg.peak_demand_mw)


def existing_hydro_mw(region: Region, cfg) -> float:
    """EEP's published fleet when available, else the hardcoded guess."""
    if DATA_AVAILABLE and cfg.use_real_plants:
        caps = D.existing_capacity_by_bus("hydro", use_available=cfg.use_available_capacity)
        if caps:
            return float(caps.get(region.name, 0.0))
    return region.hydro_mw


def existing_capacity_mw(carrier: str, region: Region, cfg) -> float:
    if DATA_AVAILABLE and cfg.use_real_plants:
        caps = D.existing_capacity_by_bus(carrier, use_available=cfg.use_available_capacity)
        return float(caps.get(region.name, 0.0))
    return 0.0


def effective_techs(cfg) -> Dict[str, Tech]:
    """TECHS with any technology-data figures merged over the defaults."""
    if not (DATA_AVAILABLE and cfg.use_real_costs):
        return TECHS
    try:
        real = D.real_tech_costs()
    except Exception as exc:
        logger.warning("technology-data costs unavailable: %s", exc)
        return TECHS
    if not real:
        return TECHS

    merged = {}
    for key, tech in TECHS.items():
        override = real.get(key, {})
        merged[key] = Tech(
            carrier=tech.carrier,
            capex=override.get("capex", tech.capex),
            fom_pct=override.get("fom_pct", tech.fom_pct),
            lifetime=override.get("lifetime", tech.lifetime),
            marginal_cost=override.get("marginal_cost", tech.marginal_cost),
            co2=tech.co2,
        )
    return merged


def transmission_corridors(cfg) -> Tuple[List[dict], dict]:
    """
    Corridors to build as Lines, plus a report of where they came from.

    Real lines aggregated by data.aggregate_corridors when the shapefile is
    present. Any bus the real data leaves disconnected gets the synthetic
    corridor(s) touching it, explicitly flagged -- an island with no
    connection would otherwise fail validation and the run would not start.
    """
    buses = pd.DataFrame({"x": [r.lon for r in REGIONS], "y": [r.lat for r in REGIONS]},
                         index=[r.name for r in REGIONS])
    report: dict = {"source": "synthetic"}

    real = None
    if DATA_AVAILABLE and cfg.use_real_grid:
        try:
            real, report = D.aggregate_corridors(buses, source=cfg.grid_source)
            report["network"] = report.get("source")          # osm | wb2007
            report["source"] = "real" if real is not None else "synthetic"
        except Exception as exc:
            logger.warning("real transmission unavailable: %s", exc)
            report = {"source": "synthetic", "status": f"error: {exc}"}

    if real is None:
        out = [{"bus0": b0, "bus1": b1, "length": km, "s_nom": mw,
                "x": 0.28 * km, "r": 0.03 * km, "s_max_pu": 1.0,
                "origin": "synthetic"}
               for b0, b1, km, mw in CORRIDORS]
        return out, report

    out = []
    for c in real.to_dict("records"):
        c["s_max_pu"] = D.S_MAX_PU
        out.append(c)

    patched = []
    for bus in report.get("isolated_buses", []):
        for b0, b1, km, mw in CORRIDORS:
            if bus in (b0, b1):
                out.append({"bus0": b0, "bus1": b1, "length": km, "s_nom": mw,
                            "x": 0.28 * km, "r": 0.03 * km, "s_max_pu": 1.0,
                            "origin": "synthetic patch"})
                patched.append(f"{b0}-{b1}")
    report["patched"] = patched
    return out, report


# --------------------------------------------------------------------------
# Network construction
# --------------------------------------------------------------------------
def _bus_frame() -> pd.DataFrame:
    return pd.DataFrame({"x": [r.lon for r in REGIONS], "y": [r.lat for r in REGIONS]},
                        index=[r.name for r in REGIONS])


def add_demand(n: pypsa.Network, sn: pd.DatetimeIndex, cfg, meta: dict) -> None:
    """
    Four load components, each its own Load so results can report them apart:

      load_<bus>     non-mining domestic, shaped, grows at the underlying rate
      access_<bus>   households connected after the base year, shaped
      mining_<bus>   flat, frozen (permit freeze) x October status, CURTAILABLE
      export_<bus>   flat, frozen, protected like domestic load

    Every load is a time series (never a scalar) so validation and KPIs,
    which read loads_t.p_set, always see all of it.
    """
    anchors = D.load_demand_anchors() if (DATA_AVAILABLE and cfg.use_real_demand) else None
    if anchors is None or "base_year" not in anchors:
        for r in REGIONS:
            n.add("Load", f"load_{r.name}", bus=r.name, carrier="AC",
                  p_set=load_profile(sn, r, cfg.peak_demand_mw))
        meta["demand"] = {"source": "synthetic"}
        return

    year = cfg.year
    comp = D.demand_components_gwh(anchors, year, cfg.mining_status, cfg.underlying_growth)
    base_year = int(anchors["base_year"]["calendar_year"])

    shares = anchors["bus_shares"]
    for r in REGIONS:
        n.add("Load", f"load_{r.name}", bus=r.name, carrier="AC",
              p_set=D.shaped_load(sn, comp["non_mining"] * shares.get(r.name, 0.0), anchors))

    acc_info = None
    if cfg.include_access:
        acc = D.load_access()
        if acc is not None:
            acc_info = D.access_demand(acc, year, base_year, cfg.grid_access_target,
                                       cfg.grid_access_target_year, cfg.kwh_per_new_household)
            weights, wrep = D.access_bus_weights(_bus_frame(), acc)
            acc_info["bus_weights"] = weights.round(4).to_dict()
            acc_info["allocation"] = wrep
            if acc_info["new_demand_gwh"] > 0:
                for r in REGIONS:
                    gwh = acc_info["new_demand_gwh"] * float(weights.get(r.name, 0.0))
                    if gwh > 0:
                        n.add("Load", f"access_{r.name}", bus=r.name, carrier="AC",
                              p_set=D.shaped_load(sn, gwh, anchors))

    mining = anchors["mining"]
    for bus, share in mining["bus_shares"].items():
        gwh = comp["mining"] * float(share)
        if gwh <= 0:
            continue
        p = D.flat_load(sn, gwh)
        n.add("Load", f"mining_{bus}", bus=bus, carrier="AC", p_set=p)
        # curtailable: shed at roughly the tariff miners pay, long before VOLL
        n.add("Generator", f"mining_curtail_{bus}", bus=bus, carrier="mining_curtailment",
              p_nom=float(p.max()), p_nom_extendable=False,
              marginal_cost=mining_price(cfg, anchors), capital_cost=0.0)

    export_info = {}
    if "destinations" in anchors["exports"]:
        for name, bus, series, info in D.export_loads(sn, anchors, comp["exports"],
                                                      cfg.resolution_hours):
            n.add("Load", f"export_{name}", bus=bus, carrier="AC", p_set=series)
            export_info[name] = dict(info, bus=bus,
                                     window_mean_mw=round(float(series.mean()), 1))
    else:                                   # older yaml layout
        for bus, share in anchors["exports"]["bus_shares"].items():
            gwh = comp["exports"] * float(share)
            if gwh > 0:
                n.add("Load", f"export_{bus}", bus=bus, carrier="AC", p_set=D.flat_load(sn, gwh))

    # Firm loads with a sourced annual energy (e.g. the railway, once known).
    # Prefixed load_ so results count them with existing customers.
    for name, gwh, shares in D.firm_load_entries(anchors):
        for bus, share in shares.items():
            if gwh * float(share) > 0:
                n.add("Load", f"load_{_slug(name)}_{bus}", bus=bus, carrier="AC",
                      p_set=D.flat_load(sn, gwh * float(share)))

    meta["demand"] = {"source": "real", "year": year, "components_gwh": comp,
                      "mining_status": cfg.mining_status, "mining_curtail_price": mining_price(cfg, anchors), "access": acc_info,
                      "exports": export_info}


def _hydro_full_year(bus: str, year: int) -> Tuple[pd.Series, str]:
    """Hourly runoff SHAPE for the whole scenario year at a bus. Weather-year
    profiles are matched by month/day/hour, so a 2030 run uses 2013 hydrology."""
    idx = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", inclusive="left", freq="h")
    if DATA_AVAILABLE:
        cache = D.load_cached_profile("hydro")
        if cache is not None and bus in cache.columns:
            s = D.profile_for("hydro", idx, bus, cache=_PROFILE_CACHE)
            if s is not None:
                return s.clip(lower=0.0), "ERA5 runoff"
    return hydro_profile(idx).clip(lower=0.0), "synthetic"


def _window_mean(hourly: pd.Series, sn: pd.DatetimeIndex, res_h: int) -> pd.Series:
    """Average an hourly series over each snapshot's [t, t+res) interval, so
    window inflow carries exactly the energy the hourly rule curve assumes."""
    hours = (sn.repeat(res_h) + pd.to_timedelta(np.tile(np.arange(res_h), len(sn)), unit="h"))
    key = pd.MultiIndex.from_arrays([hourly.index.month, hourly.index.day, hourly.index.hour])
    lookup = pd.Series(hourly.to_numpy(), index=key)
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    want = pd.MultiIndex.from_arrays([hours.month, hours.day, hours.hour])
    vals = lookup.reindex(want).ffill().bfill().to_numpy().reshape(len(sn), res_h)
    return pd.Series(vals.mean(axis=1), index=sn)


def add_hydro_reservoirs(n: pypsa.Network, sn: pd.DatetimeIndex, cfg, meta: dict) -> bool:
    """
    Each hydro plant as a StorageUnit fed by inflow:

      inflow        ERA5 runoff SHAPE for the plant's bus, scaled so a full
                    year delivers the plant's design energy x inflow_factor
      max_hours     storage_months of design energy
      p_min_pu = 0  no pumping from the grid
      spill         automatic -- water that cannot be used or stored is spilled

    Returns False if the plant table is missing (caller falls back).
    """
    if not (DATA_AVAILABLE and cfg.use_real_plants):
        return False
    plants = D.load_hydro_plants()
    if plants is None:
        return False

    year = cfg.year
    drawdown = cfg.hydro_mode == "drawdown"
    year_fraction = len(sn) * cfg.resolution_hours / HOURS_PER_YEAR
    # A full-year run can be cyclic. A shorter window cannot: a cyclic January
    # window would forbid releasing the water stored during the Kiremt rains.
    # Windows instead start on the plant's rule curve and must end at or above it.
    full_year_run = len(sn) * cfg.resolution_hours >= HOURS_PER_YEAR - cfg.resolution_hours
    soc_floor = {}
    added = []
    for row in plants.itertuples():
        frac = D.capacity_fraction(row, year)
        if frac <= 0:
            added.append({"name": row.name, "online_mw": 0.0, "note": f"not online until {int(row.online_year)}"})
            continue
        p_nom = float(row.capacity_mw) * frac
        full_shape, shape_src = _hydro_full_year(row.bus, year)
        full_mean = float(full_shape.mean())
        if full_mean <= 0:
            continue
        avg_mw = float(row.design_gwh) * 1000.0 / HOURS_PER_YEAR * float(cfg.inflow_factor)
        inflow_hourly = full_shape / full_mean * avg_mw
        inflow = _window_mean(inflow_hourly, sn, cfg.resolution_hours).clip(lower=0.0)
        storage_mwh = float(row.storage_months) / 12.0 * float(row.design_gwh) * 1000.0
        max_hours = max(storage_mwh / p_nom, 1.0)

        # Drawdown only where there is evidence for it, and only by that much.
        allowance = float(getattr(row, "max_drawdown_gwh", 0.0) or 0.0) * 1000.0 * year_fraction
        full = p_nom * max_hours
        draws = drawdown and allowance > 0
        name = f"hydro_{_slug(row.name)}"
        # Rule curve whenever the run is not a plain cyclic full year.
        # Drawdown used to start the reservoir FULL -- wrong for a run that
        # begins in July, when Ethiopian reservoirs are at their seasonal LOW
        # and the Kiremt flood is about to arrive: a full reservoir had to
        # spill it. Now drawdown follows the same seasonal curve, starting
        # `allowance` above it: the evidence is that the reservoir held that
        # much MORE water than its steady cycle, not that it was brim-full.
        use_curve = draws or not full_year_run
        soc0 = 0.0
        if use_curve:
            curve = H.rule_curve(inflow_hourly, p_nom, full)
            start_level = H.curve_level_at(curve, sn[0] - pd.Timedelta(hours=1))
            end_level = H.curve_level_at(
                curve, sn[-1] + pd.Timedelta(hours=cfg.resolution_hours - 1))
            soc0 = min(start_level + allowance, full) if draws else start_level
        windowed = use_curve and not draws
        n.add("StorageUnit", name, bus=row.bus, carrier="hydro",
              p_nom=p_nom, p_nom_extendable=False, max_hours=max_hours,
              p_min_pu=0.0, efficiency_dispatch=1.0, efficiency_store=1.0,
              inflow=inflow, marginal_cost=0.5, capital_cost=0.0,
              cyclic_state_of_charge=not use_curve,
              state_of_charge_initial=soc0)
        if use_curve:
            soc_floor[name] = end_level
        added.append({"name": row.name, "bus": row.bus, "online_mw": p_nom,
                      "design_gwh": float(row.design_gwh), "storage_mwh": p_nom * max_hours,
                      "basis": row.energy_basis, "shape": shape_src,
                      "boundary": "rule curve + drawdown" if draws else ("rule curve" if windowed else "cyclic"),
                      "soc_start_mwh": soc0,
                      "soc_end_min_mwh": soc_floor.get(name)})
    meta["hydro"] = {"mode": cfg.hydro_mode, "inflow_factor": cfg.inflow_factor, "plants": added}
    meta["soc_floor"] = soc_floor
    return True


def calibrated_wind(sn: pd.DatetimeIndex, region: Region, factor: Optional[float]) -> pd.Series:
    base = wind_cf(sn, region)
    if factor is None:
        return base
    return (base * factor).clip(0.0, 1.0)


def add_existing_wind(n: pypsa.Network, sn: pd.DatetimeIndex, cfg, techs, meta: dict):
    """
    Existing wind farms, one Generator each, with the bus's ERA5 wind SHAPE
    scaled so its full-year average equals the farm's observed capacity
    factor. Returns {bus: factor} for the capacity-weighted farms at each bus,
    reused for new-build wind at that bus (a real farm proves good sites
    exist there). Returns None if there is no observed data to calibrate to.
    """
    if not (DATA_AVAILABLE and cfg.use_real_plants):
        return None
    plants = D.load_plants()
    if plants is None or "observed_cf" not in plants.columns:
        return None
    wind = plants[(plants.carrier == "wind") & plants.observed_cf.notna()]
    if wind.empty:
        return None

    col = "available_mw" if cfg.use_available_capacity else "capacity_mw"
    cache = D.load_cached_profile("wind") if DATA_AVAILABLE else None
    by_region = {r.name: r for r in REGIONS}
    factors, report = {}, []
    for bus, grp in wind.groupby("bus"):
        region = by_region.get(bus)
        if region is None:
            continue
        if cache is not None and bus in cache.columns:
            full = cache[bus].to_numpy()
        else:
            full_idx = pd.date_range(f"{sn[0].year}-01-01", periods=HOURS_PER_YEAR, freq="h")
            full = wind_profile(full_idx, region).to_numpy()
        target_bus = float((grp.observed_cf * grp[col]).sum() / grp[col].sum())
        factors[bus] = D.scale_to_mean(full, target_bus)
        for f in grp.itertuples():
            fac = D.scale_to_mean(full, float(f.observed_cf))
            n.add("Generator", f"wind_existing_{_slug(f.name)}", bus=bus, carrier="wind",
                  p_nom=float(getattr(f, col)), p_nom_extendable=False,
                  marginal_cost=techs["wind"].marginal_cost, capital_cost=0.0,
                  p_max_pu=calibrated_wind(sn, region, fac))
            report.append({"farm": f.name, "bus": bus, "target_cf": float(f.observed_cf),
                           "regional_cf": float(full.mean()), "factor": round(fac, 2)})
    meta["wind_calibration"] = report
    return factors


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")


def build_network(cfg: ScenarioConfig) -> pypsa.Network:
    n = pypsa.Network()

    # ---- STEP 1, ALWAYS FIRST: snapshots -------------------------------
    sn = make_snapshots(cfg)
    n.set_snapshots(sn)
    n.snapshot_weightings.loc[:, :] = float(cfg.resolution_hours)

    techs = effective_techs(cfg)
    modelled_hours = len(sn) * cfg.resolution_hours
    year_fraction = modelled_hours / HOURS_PER_YEAR   # scales annual fixed costs

    # ---- carriers ------------------------------------------------------
    n.add("Carrier", "AC")
    for key, tech in TECHS.items():
        n.add("Carrier", tech.carrier, co2_emissions=tech.co2)
    n.add("Carrier", "load_shedding")
    n.add("Carrier", "mining_curtailment")

    # ---- buses ---------------------------------------------------------
    for r in REGIONS:
        n.add("Bus", r.name, v_nom=400.0, x=r.lon, y=r.lat,
              country="ET", carrier="AC")

    # ---- transmission --------------------------------------------------
    corridors, grid_report = transmission_corridors(cfg)
    seen = set()
    for c in corridors:
        name = f"{c['bus0']}-{c['bus1']}"
        if name in seen:          # a patch may duplicate a real pair
            continue
        seen.add(name)
        km = float(c["length"])
        n.add(
            "Line", name,
            bus0=c["bus0"], bus1=c["bus1"], length=km,
            x=float(c["x"]), r=float(c["r"]),
            s_nom=float(c["s_nom"]),
            s_max_pu=float(c.get("s_max_pu", 1.0)),
            s_nom_extendable=cfg.expand_network and cfg.allow_new_build,
            s_nom_min=float(c["s_nom"]),
            s_nom_max=float(c["s_nom"]) * 4.0,
            # ~450 EUR/MW/km overnight, 40 y life
            capital_cost=450.0 * km * (annuity(40, cfg.discount_rate) + 0.01) * year_fraction,
        )

    # ---- demand --------------------------------------------------------
    meta = {"grid_report": grid_report}
    add_demand(n, sn, cfg, meta)

    # ---- existing hydro (fixed capacity, seasonal inflow) --------------
    if cfg.existing_hydro:
        if not add_hydro_reservoirs(n, sn, cfg, meta):
            # fallback: old run-of-river representation
            for r in REGIONS:
                mw = existing_hydro_mw(r, cfg)
                if mw <= 0:
                    continue
                n.add(
                    "Generator", f"hydro_existing_{r.name}",
                    bus=r.name, carrier="hydro",
                    p_nom=mw, p_nom_extendable=False,
                    marginal_cost=1.0,
                    p_max_pu=hydro_cf(sn, r),
                    capital_cost=0.0,
                )

    # ---- other existing plants from the published fleet ----------------
    wind_factors = add_existing_wind(n, sn, cfg, techs, meta)
    for r in REGIONS:
        for carrier, profile_fn in (("solar", solar_cf),) + (
                (("wind", wind_cf),) if wind_factors is None else ()):
            mw = existing_capacity_mw(carrier, r, cfg)
            if mw <= 0:
                continue
            n.add(
                "Generator", f"{carrier}_existing_{r.name}",
                bus=r.name, carrier=carrier,
                p_nom=mw, p_nom_extendable=False,
                marginal_cost=techs[carrier].marginal_cost,
                p_max_pu=profile_fn(sn, r),
                capital_cost=0.0,
            )
        for carrier in ("geothermal", "diesel"):
            mw = existing_capacity_mw(carrier, r, cfg)
            if mw <= 0:
                continue
            n.add(
                "Generator", f"{carrier}_existing_{r.name}",
                bus=r.name, carrier=carrier,
                p_nom=mw, p_nom_extendable=False,
                marginal_cost=techs[carrier].marginal_cost,
                capital_cost=0.0,
            )

    # ---- candidate new capacity ---------------------------------------
    def fixed(tech_key: str) -> float:
        return annual_fixed_cost(techs[tech_key], cfg.discount_rate) * year_fraction

    for r in (REGIONS if cfg.allow_new_build else []):
        if cfg.enabled.get("solar", True):
            n.add("Generator", f"solar_{r.name}", bus=r.name, carrier="solar",
                  p_nom_extendable=True, p_nom_max=20_000.0,
                  capital_cost=fixed("solar"),
                  marginal_cost=techs["solar"].marginal_cost,
                  p_max_pu=solar_cf(sn, r))

        if cfg.enabled.get("wind", True):
            n.add("Generator", f"wind_{r.name}", bus=r.name, carrier="wind",
                  p_nom_extendable=True, p_nom_max=10_000.0,
                  capital_cost=fixed("wind"),
                  marginal_cost=techs["wind"].marginal_cost,
                  p_max_pu=calibrated_wind(sn, r, (wind_factors or {}).get(r.name)))

        if cfg.enabled.get("hydro", True) and r.hydro_mw > 0:
            n.add("Generator", f"hydro_new_{r.name}", bus=r.name, carrier="hydro",
                  p_nom_extendable=True, p_nom_max=3_000.0,
                  capital_cost=fixed("hydro"),
                  marginal_cost=techs["hydro"].marginal_cost,
                  p_max_pu=hydro_cf(sn, r))

        if cfg.enabled.get("diesel", True):
            n.add("Generator", f"diesel_{r.name}", bus=r.name, carrier="diesel",
                  p_nom_extendable=True,
                  capital_cost=fixed("diesel"),
                  marginal_cost=techs["diesel"].marginal_cost)

        if cfg.enabled.get("battery", True):
            n.add("StorageUnit", f"battery_{r.name}", bus=r.name, carrier="battery",
                  p_nom_extendable=True, max_hours=4.0,
                  capital_cost=fixed("battery"),
                  efficiency_store=0.95, efficiency_dispatch=0.95,
                  cyclic_state_of_charge=True)

    # geothermal only in the Rift Valley nodes
    if cfg.enabled.get("geothermal", True) and cfg.allow_new_build:
        for name, cap in (("Hawassa", 2000.0), ("Addis_Ababa", 1000.0)):
            n.add("Generator", f"geothermal_{name}", bus=name, carrier="geothermal",
                  p_nom_extendable=True, p_nom_max=cap,
                  capital_cost=fixed("geothermal"),
                  marginal_cost=techs["geothermal"].marginal_cost,
                  p_max_pu=0.85)

    # ---- slack: guarantees a feasible model, priced at VOLL ------------
    # Sized from the load actually in the network, so switching to anchored
    # demand does not leave an oversized slack distorting the bounds.
    # Capped at each bus's OWN load, hour by hour. Unserved energy is negative
    # load: it cannot exceed what is there. Sized to the system peak instead,
    # a bus could "shed" more than its demand and export the surplus -- equal
    # cost to shedding where the shortage really is, so the optimiser picked
    # arbitrarily and per-bus shortfalls and line flows came out wrong.
    bus_load = pd.DataFrame(0.0, index=n.snapshots, columns=[r.name for r in REGIONS])
    for ld, bus in n.loads.bus.items():
        if ld in n.loads_t.p_set.columns:
            bus_load[bus] += n.loads_t.p_set[ld]
        else:
            bus_load[bus] += float(n.loads.at[ld, "p_set"])
    # Tiered, like rotational load shedding: each bus's load is split into five
    # 20% blocks priced 1% apart. With a single price, every split of a regional
    # shortfall costs the same, and the solver picked an extreme (Jigjiga fully
    # dark while Dire Dawa barely shed). Tiers make shallow cuts everywhere
    # cheaper than a deep cut anywhere, so shortfalls spread proportionally.
    tiers = 5
    for r in REGIONS:
        peak = float(bus_load[r.name].max())
        if peak <= 0:
            continue
        pu = (bus_load[r.name] / peak).clip(0.0, 1.0)
        for k in range(tiers):
            n.add("Generator", f"shed_{r.name}_t{k}", bus=r.name, carrier="load_shedding",
                  p_nom=peak / tiers, p_nom_extendable=False, p_max_pu=pu,
                  marginal_cost=VOLL * (1.0 + 0.01 * k), capital_cost=0.0)

    n.meta = dict(getattr(n, "meta", {}) or {}, **meta)

    # ---- CO2 cap -------------------------------------------------------
    if cfg.co2_limit_t is not None:
        n.add("GlobalConstraint", "co2_limit",
              type="primary_energy", carrier_attribute="co2_emissions",
              sense="<=", constant=float(cfg.co2_limit_t) * year_fraction)

    return n


# --------------------------------------------------------------------------
# Pre-flight validation -- catches the class of bug that broke the old app
# --------------------------------------------------------------------------
def validate_network(n: pypsa.Network) -> List[Tuple[str, bool, str]]:
    """Returns (check name, passed, detail)."""
    checks: List[Tuple[str, bool, str]] = []

    n_snap = len(n.snapshots)
    checks.append(("Snapshots set", n_snap > 1, f"{n_snap} snapshots"))

    demand = float(n.loads_t.p_set.sum().sum()) if not n.loads_t.p_set.empty else 0.0
    weighted = demand * float(n.snapshot_weightings.objective.mean() or 1.0)
    checks.append(("Demand is non-zero", demand > 0, f"{weighted:,.0f} MWh over the horizon"))

    aligned = (not n.loads_t.p_set.empty) and len(n.loads_t.p_set) == n_snap
    checks.append(("Load profiles aligned to snapshots", aligned,
                   f"{len(n.loads_t.p_set)} rows vs {n_snap} snapshots"))

    ext = int(n.generators.p_nom_extendable.sum())
    fixed_cap = float(n.generators.loc[~n.generators.p_nom_extendable, "p_nom"].sum())
    checks.append(("Supply available", ext > 0 or fixed_cap > 0,
                   f"{ext} extendable generators, {fixed_cap:,.0f} MW fixed"))

    if "p_max_pu" in n.generators_t and not n.generators_t.p_max_pu.empty:
        dead = [c for c in n.generators_t.p_max_pu.columns
                if float(n.generators_t.p_max_pu[c].sum()) <= 0]
        checks.append(("No all-zero availability profiles", not dead,
                       "all profiles carry energy" if not dead else f"zero: {', '.join(dead)}"))
    else:
        checks.append(("No all-zero availability profiles", True,
                       "no time-varying profiles to check"))

    isolated = [b for b in n.buses.index
                if b not in set(n.lines.bus0) | set(n.lines.bus1)]
    checks.append(("All buses connected", not isolated,
                   "connected" if not isolated else f"isolated: {', '.join(isolated)}"))

    costed = int((n.generators.capital_cost > 0).sum()) + int((n.generators.marginal_cost > 0).sum())
    checks.append(("Costs assigned", costed > 0, f"{costed} generator cost entries"))

    return checks


# --------------------------------------------------------------------------
# Solve + KPI extraction
# --------------------------------------------------------------------------
def _soc_floor_constraints(n: pypsa.Network, sns) -> None:
    """End-of-horizon floor on reservoir energy for plants allowed to draw
    down. PyPSA has no SOC-bound attribute, so it is added directly."""
    floors = (getattr(n, "meta", {}) or {}).get("soc_floor", {})
    if not floors:
        return
    soc = n.model.variables["StorageUnit-state_of_charge"]
    comp_dim = [d for d in soc.dims if d != "snapshot"][0]
    for name, floor in floors.items():
        n.model.add_constraints(
            soc.sel({"snapshot": sns[-1], comp_dim: name}) >= floor,
            name=f"soc_floor_{name}")


def solve(n: pypsa.Network, solver: str = "highs") -> Tuple[str, str]:
    # Interior point without crossover: a full-year run with investment
    # solves in ~40 s instead of several minutes with simplex.
    opts = {"solver": "ipm", "run_crossover": "off"} if solver == "highs" else {}
    status, condition = n.optimize(solver_name=solver,
                                   extra_functionality=_soc_floor_constraints, **opts)
    return str(status), str(condition)


def _by_carrier(series: pd.Series) -> pd.Series:
    """Collapse a statistics MultiIndex (component, carrier) down to carrier."""
    if series.empty:
        return pd.Series(dtype=float)
    if isinstance(series.index, pd.MultiIndex):
        series = series.groupby(level=-1).sum()
    return series.sort_values(ascending=False)


RENEWABLES = {"solar", "wind", "hydro", "geothermal"}


def mining_price(cfg: ScenarioConfig, anchors: dict) -> float:
    """EUR/MWh charged for curtailing miners: the scenario override, else the yaml tariff."""
    if cfg.mining_curtail_price is not None:
        return float(cfg.mining_curtail_price)
    return float(anchors["mining"]["curtail_price_eur_mwh"])


def existing_line_cost(n: pypsa.Network) -> pd.Series:
    """
    Annualised capital cost of the transmission that already exists, by carrier.

    Lines are built with s_nom = today's capacity and a capital cost on every
    MW, so PyPSA's capex counts the existing grid as if it were new. Existing
    plants carry zero capital cost, so the two were treated inconsistently.
    This is a constant: it never changes a dispatch or build decision, only
    the reported cost. summarise() removes it so capex means new spending.
    """
    if n.lines.empty:
        return pd.Series(dtype=float)
    return (n.lines.capital_cost * n.lines.s_nom).groupby(n.lines.carrier).sum()


def summarise(n: pypsa.Network) -> dict:
    stats = n.statistics
    capex_raw = stats.capex()
    existing_grid = existing_line_cost(n)
    if isinstance(capex_raw.index, pd.MultiIndex):
        capex_raw = capex_raw.copy()
        for carrier, value in existing_grid.items():
            key = ("Line", carrier)
            if key in capex_raw.index:
                capex_raw[key] = max(float(capex_raw[key]) - float(value), 0.0)
    capex = _by_carrier(capex_raw)
    capex = capex[capex.abs() > 1e-3]
    opex = _by_carrier(stats.opex())
    w = n.snapshot_weightings.objective

    # generators by carrier, straight from dispatch
    gen_energy = n.generators_t.p.mul(w, axis=0).sum()
    by_carrier = gen_energy.groupby(n.generators.carrier).sum()

    # reservoir hydro is a StorageUnit: its discharge IS generation (the energy
    # came from inflow). Batteries are storage and are reported separately.
    storage_discharge = 0.0
    spill = 0.0
    if not n.storage_units.empty and not n.storage_units_t.p.empty:
        disp = n.storage_units_t.p.clip(lower=0).mul(w, axis=0).sum()
        su_by = disp.groupby(n.storage_units.carrier).sum()
        if "hydro" in su_by:
            by_carrier["hydro"] = by_carrier.get("hydro", 0.0) + su_by["hydro"]
        storage_discharge = float(su_by.drop(labels=["hydro"], errors="ignore").sum())
        if "spill" in n.storage_units_t and not n.storage_units_t.spill.empty:
            spill = float(n.storage_units_t.spill.mul(w, axis=0).sum().sum())

    by_carrier = by_carrier[by_carrier.abs() > 1e-6].sort_values(ascending=False)
    shed = float(by_carrier.get("load_shedding", 0.0))
    mining_curtailed = float(by_carrier.get("mining_curtailment", 0.0))
    real_gen = by_carrier.drop(labels=["load_shedding", "mining_curtailment"], errors="ignore")
    served = float(real_gen.sum())
    renewable = float(sum(real_gen.get(c, 0.0) for c in RENEWABLES))

    loads = n.loads_t.p_set.mul(w, axis=0).sum()
    demand = float(loads.sum())
    def part(prefix):
        return float(loads[[c for c in loads.index if c.startswith(prefix)]].sum())
    mining_demand = part("mining_")

    total_cost = float(capex.sum() + opex.sum())

    capacity = (n.generators.groupby("carrier").p_nom_opt.sum()
                .drop(labels=["load_shedding", "mining_curtailment"], errors="ignore"))
    if not n.storage_units.empty:
        capacity = pd.concat([capacity, n.storage_units.groupby("carrier").p_nom_opt.sum()])
        capacity = capacity.groupby(level=0).sum()
    capacity = capacity[capacity > 1e-3].sort_values(ascending=False)

    co2 = sum(float(real_gen.get(c, 0.0)) * t.co2 for c, t in TECHS.items() if t.co2 > 0)

    hours = float(w.sum())
    _pen_gens = n.generators.index[n.generators.carrier.isin(["load_shedding", "mining_curtailment"])]
    _pen_cost = ((n.generators_t.p[_pen_gens].mul(n.snapshot_weightings.objective, axis=0)
                  * n.generators.loc[_pen_gens, "marginal_cost"]).sum()
                 .groupby(n.generators.loc[_pen_gens, "carrier"]).sum()) if len(_pen_gens) else pd.Series(dtype=float)
    penalty = float(_pen_cost.sum())
    return {
        "status_ok": True,
        "total_cost_eur": total_cost,
        # Unserved load (VOLL) and mining curtailment (tariff) are PRICED to steer
        # the optimiser, not spent. Reported separately so they don't swamp costs.
        "penalty_eur": penalty,
        "voll_penalty_eur": float(_pen_cost.get("load_shedding", 0.0)),
        "mining_curtailment_eur": float(_pen_cost.get("mining_curtailment", 0.0)),
        "system_cost_eur": total_cost - penalty,
        "capex_eur": float(capex.sum()),
        # annualised value of the existing grid, EXCLUDED from the costs above
        "existing_grid_annuity_eur": float(existing_grid.sum()),
        "opex_eur": float(opex.sum()),
        "demand_mwh": demand,
        "demand_parts_mwh": {"non_mining": part("load_"), "access": part("access_"),
                             "mining": mining_demand, "exports": part("export_")},
        "generation_mwh": served,
        "storage_discharge_mwh": storage_discharge,
        "hydro_spill_mwh": spill,
        "renewable_share": (renewable / served * 100.0) if served > 0 else 0.0,
        "unserved_mwh": shed,
        "unserved_share": (shed / max(demand - mining_demand, 1e-9) * 100.0),
        "mining_curtailed_mwh": mining_curtailed,
        "mining_served_share": ((mining_demand - mining_curtailed) / mining_demand * 100.0)
                                if mining_demand > 0 else None,
        "lcoe_eur_mwh": ((total_cost - penalty) / served) if served > 0 else 0.0,
        "co2_t": co2,
        "capacity_mw": capacity,
        "supply_mwh": real_gen,
        "capex_by_carrier": capex,
        "opex_by_carrier": opex,
        "line_expansion": (n.lines.s_nom_opt - n.lines.s_nom_min).round(1),
        "horizon_hours": hours,
        "annualised_generation_gwh": served / hours * HOURS_PER_YEAR / 1000.0 if hours else 0.0,
        "meta": getattr(n, "meta", {}),
    }


def dispatch_frame(n: pypsa.Network) -> pd.DataFrame:
    """Dispatch by carrier, MW. Reservoir hydro is shown as generation;
    batteries as a signed net column."""
    gen = n.generators_t.p.T.groupby(n.generators.carrier).sum().T
    if not n.storage_units.empty and not n.storage_units_t.p.empty:
        su = n.storage_units_t.p
        hydro_cols = n.storage_units.index[n.storage_units.carrier == "hydro"]
        if len(hydro_cols):
            gen["hydro"] = gen.get("hydro", 0.0) + su[hydro_cols].clip(lower=0).sum(axis=1)
        other = n.storage_units.index[n.storage_units.carrier != "hydro"]
        if len(other):
            st = su[other].T.groupby(n.storage_units.carrier[other]).sum().T
            gen = gen.join(st.rename(columns=lambda c: f"{c}_net"), how="left")
    return gen.round(2)


EEP_BASE_YEAR_GWH = 35734   # FY2025/26, validation target


def _self_test() -> None:
    logging.disable(logging.WARNING)
    cfg = ScenarioConfig(start="2030-01-01", days=28, resolution_hours=6)
    n = build_network(cfg)

    print("validation:")
    for name, ok, detail in validate_network(n):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    status, condition = solve(n, cfg.solver)
    print(f"\nsolver: {status} / {condition}")

    s = summarise(n)
    print(f"total system cost : EUR {s['total_cost_eur']:,.0f}")
    print(f"demand            : {s['demand_mwh']:,.0f} MWh")
    print(f"generation        : {s['generation_mwh']:,.0f} MWh")
    print(f"renewable share   : {s['renewable_share']:.1f} %")
    print(f"unserved (firm)   : {s['unserved_mwh']:,.1f} MWh ({s['unserved_share']:.2f} %)")
    print(f"mining curtailed  : {s['mining_curtailed_mwh']:,.0f} MWh")
    print(f"existing grid     : EUR {s['existing_grid_annuity_eur']:,.0f} (annuity, excluded from cost)")
    print("\noptimal capacity (MW):")
    print(s["capacity_mw"].round(1).to_string())

    assert s["generation_mwh"] > 0, "zero generation -- regression"
    assert s["renewable_share"] > 0, "zero renewables -- regression"
    print("\nself-test OK")


if __name__ == "__main__":
    _self_test()
