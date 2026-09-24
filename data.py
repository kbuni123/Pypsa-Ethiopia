"""
data.py -- real-data loaders for PyPSA-Ethiopia.

Everything here answers one question: is this number real, or did I make it up?
Each loader returns its data plus a `source` string, and `provenance()`
summarises the whole model's data state so the app can show it honestly.

Nothing in this module is required. If a data file is missing, the caller falls
back to the synthetic profile in model.py and provenance() says so loudly.

Run `python setup_data.py` once to populate data/.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from timeseries import interval_mean

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
PROFILE_DIR = DATA_DIR / "profiles"
CUTOUT_PATH = DATA_DIR / "cutouts" / "ethiopia-era5.nc"
GADM_PATH = DATA_DIR / "shapes" / "gadm41_ETH.gpkg"
COSTS_PATH = DATA_DIR / "costs" / "costs_2030.csv"
PLANTS_PATH = DATA_DIR / "plants_ethiopia.csv"
DEMAND_PATH = DATA_DIR / "demand_ethiopia.yaml"

HOURS_PER_YEAR = 8760


# --------------------------------------------------------------------------
@dataclass
class Sourced:
    """A value plus where it came from."""
    value: object
    source: str
    is_real: bool


# --------------------------------------------------------------------------
# Power plants
# --------------------------------------------------------------------------
def load_plants(path: Path = PLANTS_PATH) -> Optional[pd.DataFrame]:
    """EEP's published plant list. Returns None if the file is absent."""
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    df["available_mw"] = pd.to_numeric(df["available_mw"], errors="coerce")
    df["available_mw"] = df["available_mw"].fillna(df["capacity_mw"])
    return df.dropna(subset=["capacity_mw"])


def existing_capacity_by_bus(
    carrier: str, use_available: bool = True, path: Path = PLANTS_PATH
) -> Dict[str, float]:
    """MW of an existing carrier per bus, from the real plant list."""
    plants = load_plants(path)
    if plants is None:
        return {}
    col = "available_mw" if use_available else "capacity_mw"
    sel = plants[plants.carrier == carrier]
    return sel.groupby("bus")[col].sum().to_dict()


# --------------------------------------------------------------------------
# Demand anchors
# --------------------------------------------------------------------------
def load_demand_anchors(path: Optional[Path] = None) -> Optional[dict]:
    path = path or DEMAND_PATH
    if not path.exists():
        return None
    with open(path) as fh:
        return yaml.safe_load(fh)


def _shape_series(idx: pd.DatetimeIndex, s: dict) -> np.ndarray:
    hour = idx.hour.to_numpy()
    shape = (
        s["base"]
        + s["midday_peak_weight"]
        * np.exp(-((hour - s["midday_peak_hour"]) ** 2) / (2 * s["midday_peak_width"] ** 2))
        + s["evening_peak_weight"]
        * np.exp(-((hour - s["evening_peak_hour"]) ** 2) / (2 * s["evening_peak_width"] ** 2))
    )
    weekend = np.where(idx.dayofweek.to_numpy() >= 5, s["weekend_factor"], 1.0)
    doy = idx.dayofyear.to_numpy()
    seasonal = 1.0 + s["seasonal_amplitude"] * np.cos(2 * np.pi * (doy - 120) / 365.0)
    return shape * weekend * seasonal


def _full_year_shape_mean(s: dict, year: int) -> float:
    full = pd.date_range(f"{year}-01-01", periods=HOURS_PER_YEAR, freq="h")
    return float(_shape_series(full, s).mean())


def demand_components_gwh(anchors: dict, year: int, mining_status: str = "continue",
                          underlying_rate: Optional[float] = None) -> Dict[str, float]:
    """
    Annual energy (GWh) of each demand component for a scenario year, before
    electrification. Access-driven demand is added separately by the access
    module so the two can never double count.

      non_mining  grows at the underlying rate from the base year
      mining      frozen at base-year level (no new permits) x October status
      exports     frozen at base-year level
    """
    b = anchors["base_year"]
    total = float(b["total_generation_gwh"])
    domestic = total * float(b["domestic_share"])
    exports = total * float(b["export_share"])
    mining_base = domestic * float(anchors["mining"]["share_of_domestic"])
    non_mining_base = domestic - mining_base

    rate = anchors["growth"]["underlying_rate"] if underlying_rate is None else underlying_rate
    years = max(0, year - int(b["calendar_year"]))
    status = anchors["mining"]["status_multiplier"].get(mining_status, 1.0)
    return {
        "non_mining": non_mining_base * (1.0 + float(rate)) ** years,
        "mining": mining_base * float(status),
        "mining_base": mining_base,
        "exports": exports,
    }


@interval_mean
def shaped_load(sn: pd.DatetimeIndex, annual_gwh: float, anchors: dict) -> pd.Series:
    """Non-mining profile carrying `annual_gwh` per year, scaled on a FULL
    year's shape mean so a January window and a July window agree on the
    annual total."""
    s = anchors["shape"]
    avg_mw = annual_gwh * 1000.0 / HOURS_PER_YEAR
    scale = avg_mw / _full_year_shape_mean(s, int(sn[0].year))
    return pd.Series(_shape_series(sn, s) * scale, index=sn)


def flat_load(sn: pd.DatetimeIndex, annual_gwh: float) -> pd.Series:
    return pd.Series(annual_gwh * 1000.0 / HOURS_PER_YEAR, index=sn, dtype=float)


def implied_peak_mw(anchors: dict, year: Optional[int] = None) -> float:
    """System peak implied by the base-year anchor: shaped non-mining peak
    plus flat mining and exports."""
    year = year or int(anchors["base_year"]["calendar_year"])
    comp = demand_components_gwh(anchors, year)
    s = anchors["shape"]
    full = pd.date_range(f"{year}-01-01", periods=HOURS_PER_YEAR, freq="h")
    shape = _shape_series(full, s)
    non_mining_peak = comp["non_mining"] * 1000.0 / HOURS_PER_YEAR * shape.max() / shape.mean()
    flat = (comp["mining"] + comp["exports"]) * 1000.0 / HOURS_PER_YEAR
    return float(non_mining_peak + flat)


def implied_load_factor(anchors: dict, year: Optional[int] = None) -> float:
    year = year or int(anchors["base_year"]["calendar_year"])
    comp = demand_components_gwh(anchors, year)
    avg = (comp["non_mining"] + comp["mining"] + comp["exports"]) * 1000.0 / HOURS_PER_YEAR
    return float(avg / implied_peak_mw(anchors, year))


# Backwards-compatible alias
def derived_peak_mw(anchors: dict) -> float:
    return implied_peak_mw(anchors)



# --------------------------------------------------------------------------
# Exports by destination
# --------------------------------------------------------------------------
def _contract_mw(hours: pd.DatetimeIndex, dest: dict) -> np.ndarray:
    """Hourly MW under a destination's contracts (latest contract in force)."""
    off = np.isin(hours.hour, list(dest.get("offpeak_hours", [])))
    out = np.zeros(len(hours))
    for c in sorted(dest["contracts"], key=lambda c: str(c["from"])):
        active = hours >= pd.Timestamp(str(c["from"]))
        out = np.where(active, np.where(off, float(c["offpeak_mw"]), float(c["peak_mw"])), out)
    return out


def contract_energy_gwh(dest: dict, start, end) -> float:
    hours = pd.date_range(str(start), str(end), inclusive="left", freq="h")
    return float(_contract_mw(hours, dest).sum()) / 1000.0


def export_loads(sn: pd.DatetimeIndex, anchors: dict, exports_total_gwh: float, res_h: int):
    """
    [(name, bus, p_set Series, info)] for every export destination.

    Contract destinations follow their peak/off-peak schedule, averaged onto
    the snapshots. The residual destination carries the base-year export total
    minus the contracted destinations' base-year volume, as a flat load.
    """
    b = anchors["base_year"]
    dests = anchors["exports"]["destinations"]
    hours = sn.repeat(res_h) + pd.to_timedelta(np.tile(np.arange(res_h), len(sn)), unit="h")
    contracted_base = sum(contract_energy_gwh(d, b["fy_start"], b["fy_end"])
                          for d in dests if d.get("contracts"))
    out = []
    for d in dests:
        if d.get("contracts"):
            mw = _contract_mw(hours, d).reshape(len(sn), res_h).mean(axis=1)
            series = pd.Series(mw, index=sn)
            info = {"basis": "contract",
                    "base_year_gwh": contract_energy_gwh(d, b["fy_start"], b["fy_end"])}
        elif d.get("residual"):
            gwh = max(exports_total_gwh - contracted_base, 0.0)
            series = flat_load(sn, gwh)
            info = {"basis": "residual", "annual_gwh": gwh}
        else:
            continue
        out.append((d["name"], d["bus"], series, info))
    return out


def firm_load_entries(anchors: dict):
    """Firm loads with a stated annual energy; entries left null are skipped."""
    entries = []
    for fl in anchors.get("firm_loads", []) or []:
        gwh = fl.get("annual_gwh")
        if gwh in (None, ""):
            continue
        entries.append((str(fl["name"]), float(gwh), fl.get("bus_shares", {})))
    return entries

# --------------------------------------------------------------------------
# Hydro fleet
# --------------------------------------------------------------------------
HYDRO_PATH = DATA_DIR / "hydro_plants.csv"


def load_hydro_plants(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    path = path or HYDRO_PATH
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    for col in ("capacity_mw", "design_gwh", "storage_months", "online_year",
                "full_year", "first_fraction"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["capacity_mw", "design_gwh"])


def capacity_fraction(row, year: int) -> float:
    """Share of nameplate online in `year`: 0 before online_year, then
    first_fraction, then 1 from full_year."""
    if year < int(row.online_year):
        return 0.0
    if year >= int(row.full_year):
        return 1.0
    return float(row.first_fraction)


# --------------------------------------------------------------------------
# Wind calibration
# --------------------------------------------------------------------------
def scale_to_mean(series: np.ndarray, target: float, cap: float = 1.0) -> float:
    """Factor f such that mean(clip(f * series, 0, cap)) == target.

    A plain ratio undershoots once values start clipping at 1.0, which is
    exactly what happens when a 0.02 regional average is lifted to a 0.30
    site. Bisection on the clipped mean hits the target properly."""
    s = np.asarray(series, dtype=float)
    if s.mean() <= 0 or target <= 0:
        return 0.0
    if np.clip(s * 1e6, 0, cap).mean() < target:     # unreachable target
        return 1e6
    lo, hi = 0.0, 1.0
    while np.clip(s * hi, 0, cap).mean() < target:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.clip(s * mid, 0, cap).mean() < target:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------
# Electricity access -> new grid demand
# --------------------------------------------------------------------------
ACCESS_PATH = DATA_DIR / "access_ethiopia.yaml"
POPULATION_PATH = DATA_DIR / "population_regions.csv"


def load_access(path: Optional[Path] = None) -> Optional[dict]:
    path = path or ACCESS_PATH
    if not path.exists():
        return None
    with open(path) as fh:
        return yaml.safe_load(fh)


def grid_access_share(access: dict, year: int, target: Optional[float] = None,
                      target_year: Optional[int] = None) -> float:
    t = access["trajectory"]
    y0, a0 = int(t["grid_start_year"]), float(t["grid_start"])
    y1 = int(target_year if target_year is not None else t["grid_target_year"])
    a1 = float(target if target is not None else t["grid_target"])
    if year <= y0:
        return a0
    if year >= y1:
        return a1
    return a0 + (a1 - a0) * (year - y0) / (y1 - y0)


def households(access: dict, year: int) -> float:
    pop = float(access["national_population_2023"]) * (
        1.0 + float(access["population_growth"])) ** (year - int(access["population_base_year"]))
    return pop / float(access["household_size"])


def access_demand(access: dict, year: int, base_year: int,
                  target: Optional[float] = None, target_year: Optional[int] = None,
                  kwh_per_household: Optional[float] = None) -> dict:
    """
    New grid demand from households connected AFTER the demand anchor's base
    year. The anchor already contains everyone connected up to then, so only
    the increment counts -- no double counting.
    """
    kwh = float(kwh_per_household or access["kwh_per_new_household"])
    grid_now = grid_access_share(access, year, target, target_year)
    grid_base = grid_access_share(access, base_year, target, target_year)
    hh_now, hh_base = households(access, year), households(access, base_year)
    new_hh = max(0.0, grid_now * hh_now - grid_base * hh_base)
    out = {
        "year": year,
        "grid_access": grid_now,
        "grid_access_base": grid_base,
        "households_total": hh_now,
        "new_grid_households": new_hh,
        "new_demand_gwh": new_hh * kwh / 1e6,
        "kwh_per_household": kwh,
    }
    cost = access.get("cost_per_household_usd")
    out["connection_capex_usd"] = new_hh * float(cost) if cost else None
    # What the trajectory demands per year, next to what is actually happening.
    yrs = max(1, year - base_year)
    out["implied_connections_per_year"] = new_hh / yrs if year > base_year else 0.0
    pace = access.get("pace_reference", {}) or {}
    out["actual_connections_per_year"] = pace.get("actual_connections_per_year")
    out["target_connections_per_year"] = pace.get("target_connections_per_year")
    return out


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalpha())


def access_bus_weights(buses: pd.DataFrame, access: Optional[dict] = None):
    """
    Share of NOT-YET-CONNECTED population in each bus region -- where new grid
    connections will land.

    Regional population x (1 - regional grid access), spread uniformly over
    each administrative region's area, then intersected with the Voronoi bus
    regions. Addis Ababa's grid access is known (93%); every other region gets
    the uniform residual that reproduces the national 29.3%.

    Returns (weights Series summing to 1, report dict).
    """
    import geopandas as gpd

    access = access or load_access()
    report: dict = {"method": "population x unconnected share, area-overlap"}
    pop = pd.read_csv(POPULATION_PATH, comment="#") if POPULATION_PATH.exists() else None

    if access is None or pop is None or not GADM_PATH.exists():
        report["method"] = "fallback: non-mining demand bus shares (population data or shapes missing)"
        anchors = load_demand_anchors()
        w = pd.Series(anchors["bus_shares"]) if anchors else pd.Series(1.0, index=buses.index)
        return (w / w.sum()).reindex(buses.index).fillna(0.0), report

    metric = 32637
    gadm = gpd.read_file(GADM_PATH, layer="ADM_ADM_1").to_crs(metric)
    name_col = "NAME_1" if "NAME_1" in gadm.columns else gadm.columns[0]
    gadm["key"] = gadm[name_col].map(_norm)
    gadm["area"] = gadm.geometry.area

    # uniform residual grid access outside the known regions
    national = float(access["trajectory"]["grid_start"])
    known = access.get("known_grid_access", {}) or {}
    total_pop = pop.population.sum()
    known_pop = known_conn = 0.0
    for _, row in pop.iterrows():
        aliases = [a.strip() for a in str(row.aliases).split(";")]
        for alias, share in known.items():
            if alias in aliases:
                known_pop += row.population
                known_conn += row.population * float(share)
    residual = (national * total_pop - known_conn) / max(total_pop - known_pop, 1.0)
    residual = float(np.clip(residual, 0.0, 1.0))
    report["residual_grid_access"] = residual

    # unconnected population per GADM polygon
    gadm["unconnected"] = 0.0
    unmatched = []
    for _, row in pop.iterrows():
        aliases = [a.strip() for a in str(row.aliases).split(";") if a.strip()]
        hits = gadm.index[gadm.key.map(lambda k: any(k.startswith(a) for a in aliases))]
        if len(hits) == 0:
            unmatched.append(row.region)
            continue
        share = next((float(known[a]) for a in aliases if a in known), residual)
        uncon = row.population * (1.0 - share)
        area = gadm.loc[hits, "area"]
        gadm.loc[hits, "unconnected"] += uncon * area / area.sum()
    report["unmatched_regions"] = unmatched

    regions = bus_regions(buses).to_crs(metric)
    weights = pd.Series(0.0, index=buses.index)
    for _, g in gadm.iterrows():
        if g.unconnected <= 0:
            continue
        for bus, cell in regions.geometry.items():
            overlap = g.geometry.intersection(cell).area
            if overlap > 0:
                weights[bus] += g.unconnected * overlap / g.area

    if weights.sum() <= 0:
        report["method"] = "fallback: no overlap computed"
        weights = pd.Series(1.0, index=buses.index)
    report["unconnected_population"] = float(weights.sum())
    return weights / weights.sum(), report


# --------------------------------------------------------------------------
# Costs from technology-data
# --------------------------------------------------------------------------
# technology-data's own naming -> our carrier keys
COST_TECH_MAP = {
    "solar": "solar-utility",
    "wind": "onwind",
    "hydro": "hydro",
    "geothermal": "geothermal",
    "diesel": "oil",             # technology-data "oil" = diesel engine farm
    "battery": "battery storage",
}

# The model's battery is 4 hours of storage behind an inverter (model.TECHS).
BATTERY_HOURS = 4.0

# technology-data quotes investment per kW (or per kWh for battery storage);
# the model works per MW. Multiply by these to convert.
_PER_MW = {"EUR/kW": 1000.0, "EUR/kW_e": 1000.0, "EUR/kWel": 1000.0,
           "EUR/kWh": 1000.0, "EUR/MW": 1.0, "EUR/MWh": 1.0}


def load_costs(path: Path = COSTS_PATH) -> Optional[pd.DataFrame]:
    """The costs.csv that PyPSA-Earth's retrieve_cost_data produces."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    expected = {"technology", "parameter", "value", "unit"}
    if not expected.issubset(df.columns):
        warnings.warn(f"{path} does not look like a technology-data costs.csv")
        return None
    return df


def cost_lookup(costs: pd.DataFrame, technology: str, parameter: str) -> Optional[float]:
    sel = costs[(costs.technology == technology) & (costs.parameter == parameter)]
    if sel.empty:
        return None
    return float(sel.value.iloc[0])


def cost_unit(costs: pd.DataFrame, technology: str, parameter: str) -> Optional[str]:
    sel = costs[(costs.technology == technology) & (costs.parameter == parameter)]
    if sel.empty:
        return None
    return str(sel.unit.iloc[0]).strip()


def investment_per_mw(costs: pd.DataFrame, technology: str) -> Optional[float]:
    """Investment converted to EUR/MW (or EUR/MWh for storage energy)."""
    value = cost_lookup(costs, technology, "investment")
    if value is None:
        return None
    unit = cost_unit(costs, technology, "investment")
    if unit not in _PER_MW:
        warnings.warn(f"technology-data {technology}: unknown investment unit {unit!r}, skipped")
        return None
    return value * _PER_MW[unit]


def real_tech_costs(path: Path = COSTS_PATH) -> Dict[str, dict]:
    """
    Pull investment / FOM / VOM / lifetime per carrier out of technology-data.

    Returns {} if the file is missing, so the caller keeps its own defaults.
    Any individual field that is absent is simply omitted -- the caller merges
    over its defaults rather than getting a None.

    UNITS: technology-data quotes investment per kW. Earlier versions of this
    function passed that straight through as EUR/MW, making every new plant
    1,000x too cheap. Values are now converted using the file's unit column.
    """
    costs = load_costs(path)
    if costs is None:
        return {}

    out: Dict[str, dict] = {}
    for carrier, td_name in COST_TECH_MAP.items():
        entry = {}
        capex = investment_per_mw(costs, td_name)
        if carrier == "battery" and capex is not None:
            # EUR/MWh of storage x 4 h, plus the inverter per MW of power
            capex *= BATTERY_HOURS
            inverter = investment_per_mw(costs, "battery inverter")
            if inverter is not None:
                capex += inverter
        if capex is not None:
            entry["capex"] = capex                     # EUR/MW
        fom = cost_lookup(costs, td_name, "FOM")
        if fom is not None:
            entry["fom_pct"] = fom / 100.0             # stored as %/yr
        lifetime = cost_lookup(costs, td_name, "lifetime")
        if lifetime is not None:
            entry["lifetime"] = int(lifetime)
        if carrier == "battery":
            # storage lasts 25 y, the inverter 10 y: keep the model's 15 y
            entry.pop("lifetime", None)
        vom = cost_lookup(costs, td_name, "VOM")
        fuel = cost_lookup(costs, td_name, "fuel")
        eff = cost_lookup(costs, td_name, "efficiency")
        if fuel is not None and eff:
            # thermal plants: fuel per MWh electric plus variable O&M
            entry["marginal_cost"] = (vom or 0.0) + fuel / eff
        elif vom is not None:
            entry["marginal_cost"] = vom
        if entry:
            entry["source"] = f"technology-data: {td_name}"
            out[carrier] = entry
    return out


# --------------------------------------------------------------------------
# Weather profiles from an atlite cutout
# --------------------------------------------------------------------------
def _profile_cache_path(kind: str) -> Path:
    return PROFILE_DIR / f"{kind}.csv"


def _profile_meta_path() -> Path:
    return PROFILE_DIR / "meta.json"


def cached_profiles_stale() -> bool:
    """True if cached profiles exist but were built with an older region
    method. They must then be rebuilt, not used."""
    import json
    if not PROFILE_DIR.exists() or not any(PROFILE_DIR.glob("*.csv")):
        return False
    meta = _profile_meta_path()
    if not meta.exists():
        return True
    try:
        return json.loads(meta.read_text()).get("regions") != REGION_METHOD
    except Exception:
        return True


def load_cached_profile(kind: str) -> Optional[pd.DataFrame]:
    """
    Hourly per-unit profile, one column per bus, from the cache written by
    setup_data.py. `kind` is one of solar / wind / hydro.

    Returns None if the cache was built with an outdated region method.
    """
    path = _profile_cache_path(kind)
    if not path.exists() or cached_profiles_stale():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        return None
    return df


def cutout_available() -> bool:
    return CUTOUT_PATH.exists()


# Bump this whenever the way bus regions are drawn changes. Cached profiles
# record which method built them, and are treated as absent if it differs, so
# stale capacity factors can never be used silently.
REGION_METHOD = "voronoi-v1"


def bus_regions(buses: pd.DataFrame):
    """
    One region per bus: every point in Ethiopia belongs to its NEAREST bus.

    Voronoi cells around the bus coordinates, clipped to the national border
    taken from GADM. This replaces an earlier approach that handed whole
    administrative regions to the bus nearest each region's centroid. That
    broke badly for Ethiopia: Oromia wraps around Addis Ababa, so the Addis
    bus was left with only the ~1,000 km2 city while its whole hinterland went
    to Hawassa, and the Awash substation (in Afar) landed on Mekelle, 500+ km
    away. It also matches PyPSA-Earth's default clustering.

    `buses` needs columns x (lon) and y (lat), indexed by bus name.
    """
    import geopandas as gpd
    from shapely.geometry import MultiPoint, Point
    from shapely.ops import voronoi_diagram
    try:
        from shapely import union_all
    except ImportError:                       # shapely < 2
        from shapely.ops import unary_union as union_all

    if not GADM_PATH.exists():
        raise FileNotFoundError(
            f"{GADM_PATH} not found. Run `python setup_data.py --shapes` first."
        )

    metric = 32637   # UTM 37N: fair distances across Ethiopia
    gadm = gpd.read_file(GADM_PATH, layer="ADM_ADM_1").to_crs(metric)
    country = union_all(list(gadm.geometry)).buffer(0)

    points = gpd.GeoSeries(
        [Point(row.x, row.y) for row in buses.itertuples()],
        index=buses.index, crs=4326,
    ).to_crs(metric)

    cells = voronoi_diagram(
        MultiPoint(list(points)), envelope=country.envelope.buffer(300_000)
    )

    rows = []
    for cell in cells.geoms:
        owner = points.index[points.within(cell)]
        if len(owner) != 1:
            continue
        clipped = cell.intersection(country)
        if not clipped.is_empty:
            rows.append({"bus": owner[0], "geometry": clipped})

    regions = gpd.GeoDataFrame(rows, crs=metric).dissolve(by="bus")[["geometry"]]
    missing = set(buses.index) - set(regions.index)
    if missing:
        raise RuntimeError(f"no region could be drawn for: {sorted(missing)}")
    return regions.to_crs(4326)


def compute_profiles_from_cutout(
    buses: pd.DataFrame,
    turbine: str = "Vestas_V112_3MW",
    panel: str = "CSi",
):
    """
    Area-weighted per-unit capacity factors per bus, straight from atlite.

    Returns (profiles, failures). Each profile is computed and SAVED
    independently, so a failure in one (e.g. runoff needing a feature the
    cutout lacks) never discards the others that already succeeded.

    NOTE ON RIGOUR: this averages over each region's full land area with a
    uniform layout. It does NOT apply land-use exclusions (protected areas,
    slope, settlement buffers), which PyPSA-Earth's build_renewable_profiles
    does via atlite.ExclusionContainer and the Copernicus land-cover raster.
    Exclusions matter far more for installable potential (p_nom_max) than for
    the shape of the capacity factor, but the capacity factors here will still
    run slightly optimistic because unbuildable terrain is included in the
    average.
    """
    import atlite

    cutout = atlite.Cutout(str(CUTOUT_PATH))
    regions = bus_regions(buses)

    def solar():
        return _to_frame(cutout.pv(
            panel=panel, orientation="latitude_optimal",
            shapes=regions, per_unit=True, aggregate_time=None,
        ), regions.index)

    def wind():
        return _to_frame(cutout.wind(
            turbine=turbine,
            shapes=regions, per_unit=True, aggregate_time=None,
        ), regions.index)

    def hydro():
        # Runoff is a proxy for hydro inflow availability. It is NOT a
        # reservoir model -- no storage, no cascade, no operating rules. Good
        # enough to give hydro a real seasonal signature instead of a sine.
        df = _to_frame(cutout.runoff(
            shapes=regions, per_unit=True, aggregate_time=None,
        ), regions.index)
        return df.div(df.max().replace(0, np.nan), axis=1).fillna(0.0).clip(0, 1)

    profiles: Dict[str, pd.DataFrame] = {}
    failures: Dict[str, str] = {}
    for kind, fn in (("solar", solar), ("wind", wind), ("hydro", hydro)):
        try:
            df = fn()
            save_profiles({kind: df})          # persist immediately
            profiles[kind] = df
        except Exception as exc:
            failures[kind] = f"{type(exc).__name__}: {exc}"
            logger.warning("profile %s failed: %s", kind, exc)
    return profiles, failures


def _to_frame(da, index) -> pd.DataFrame:
    """xarray DataArray (time x shape) -> DataFrame with bus columns."""
    df = da.to_pandas()
    if not isinstance(df, pd.DataFrame):
        df = df.to_frame()
    # atlite names the shape dimension after the GeoDataFrame index
    if df.shape[0] == len(index) and df.shape[1] != len(index):
        df = df.T
    df.columns = list(index)
    df.index = pd.to_datetime(df.index)
    df.index.name = "time"
    return df.clip(0.0, 1.0)


def save_profiles(profiles: Dict[str, pd.DataFrame]) -> List[Path]:
    import json
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    _profile_meta_path().write_text(json.dumps({"regions": REGION_METHOD}))
    written = []
    for kind, df in profiles.items():
        path = _profile_cache_path(kind)
        df.to_csv(path)
        written.append(path)
    return written


# --------------------------------------------------------------------------
# Reindexing a cached profile onto model snapshots
# --------------------------------------------------------------------------
@interval_mean
def profile_for(
    kind: str, sn: pd.DatetimeIndex, bus: str, cache: Optional[dict] = None
) -> Optional[pd.Series]:
    """
    Per-unit profile for one bus over the model's snapshots, or None if the
    cached data cannot cover the request.

    Weather years rarely match scenario years, so the month-day-hour of each
    snapshot is matched against the cached year rather than the absolute
    timestamp. A 2030 scenario can therefore run on 2013 weather, which is
    exactly what PyPSA-Earth does.
    """
    store = cache if cache is not None else {}
    if kind not in store:
        store[kind] = load_cached_profile(kind)
    df = store[kind]
    if df is None or bus not in df.columns:
        return None

    series = df[bus]
    key = pd.MultiIndex.from_arrays(
        [series.index.month, series.index.day, series.index.hour]
    )
    lookup = pd.Series(series.to_numpy(), index=key)
    lookup = lookup[~lookup.index.duplicated(keep="first")]

    want = pd.MultiIndex.from_arrays([sn.month, sn.day, sn.hour])
    values = lookup.reindex(want)
    if values.isna().all():
        return None
    values = values.ffill().bfill()
    return pd.Series(values.to_numpy(), index=sn).clip(0.0, 1.0)


# --------------------------------------------------------------------------
# Provenance -- what is real in this run and what is not
# --------------------------------------------------------------------------
def provenance() -> pd.DataFrame:
    rows = []

    def add(item, real, detail):
        rows.append({"Input": item, "": "real" if real else "synthetic", "Detail": detail})

    for kind, label in (("solar", "Solar profiles"),
                        ("wind", "Wind profiles"),
                        ("hydro", "Hydro inflow")):
        df = load_cached_profile(kind)
        if df is not None:
            add(label, True,
                f"atlite/ERA5, {len(df.columns)} buses, {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
        elif cached_profiles_stale():
            add(label, False,
                "cached profile built with OLD bus regions -- run setup_data.py --profiles")
        else:
            add(label, False, "no cached profile -- using the synthetic fallback")

    plants = load_plants()
    if plants is not None:
        total = plants.available_mw.sum()
        add("Existing capacity", True,
            f"{len(plants)} plants, {total:,.0f} MW available (EEP published list)")
    else:
        add("Existing capacity", False, "hardcoded values in model.REGIONS")

    anchors = load_demand_anchors()
    if anchors is not None and "base_year" in anchors:
        b = anchors["base_year"]
        add("Demand -- annual energy", True,
            f"{b['total_generation_gwh']:,} GWh total, FY{b['fiscal_year']} (EEP)")
        add("Demand -- mining share", True,
            f"{anchors['mining']['share_of_domestic']:.0%} of domestic (EEP FY2024/25)")
        add("Demand -- mining location", False,
            "90/10 Addis/Hawassa inferred from known sites")
        add("Demand -- export split", False,
            "Kenya/Djibouti split not published; 65/35 assumed")
        add("Demand -- hourly shape", False,
            "assumed; no public hourly Ethiopian load series exists")
        add("Demand -- underlying growth", False,
            f"{anchors['growth']['underlying_rate']:.0%}/yr placeholder, no sourced figure")
    else:
        add("Demand", False, "slider-driven peak with an invented shape")

    hydro = load_hydro_plants()
    if hydro is not None:
        floors = int((hydro.energy_basis == "observed_2016_17").sum())
        add("Hydro design energy", True,
            f"{len(hydro)} plants; {len(hydro) - floors} design/derived, {floors} floors from 2016/17")
        add("Hydro reservoir size", False,
            "storage months in tiers from live volume -- heads not all published")

    access = load_access()
    if access is not None:
        add("Access -- current", True,
            f"{access['survey_2025']['grid_households']:.1%} of households on grid (Energy Access Survey 2025)")
        add("Access -- trajectory", False,
            f"grid to {access['trajectory']['grid_target']:.0%} by "
            f"{access['trajectory']['grid_target_year']} assumed; NEP 3.0 split unpublished")
        add("Access -- use per household", False,
            f"{access['kwh_per_new_household']} kWh/yr, MTF Tier 3 threshold assumed")

    costs = real_tech_costs()
    if costs:
        add("Technology costs", True,
            f"technology-data for {', '.join(sorted(costs))}")
    else:
        add("Technology costs", False, "generic 2030 figures in model.TECHS")

    if resolve_grid_source() == "osm":
        rows.append({
            "Input": "Transmission topology", "": "real",
            "Detail": "OpenStreetMap, surveyed routes incl. GERD and railway-era lines "
                      "(osm_grid.py); OSM completeness is uneven",
        })
        rows.append({
            "Input": "Transmission capacity", "": "synthetic",
            "Detail": "MW estimated from voltage class; untagged lines counted as one circuit",
        })
    elif TRANSMISSION_PATH.exists():
        rows.append({
            "Input": "Transmission topology", "": "real",
            "Detail": "World Bank / EEPCo lines, 2006-07 sources -- predates GERD, "
                      "Gibe III and Genale Dawa III",
        })
        rows.append({
            "Input": "Transmission capacity", "": "synthetic",
            "Detail": "MW estimated from voltage class; single circuit assumed",
        })
    else:
        add("Transmission corridors", False,
            "six invented lines between city pairs -- add the World Bank shapefile")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(provenance().to_string(index=False))


# --------------------------------------------------------------------------
# Transmission: real lines aggregated into inter-regional corridors
# --------------------------------------------------------------------------
TRANSMISSION_PATH = DATA_DIR / "shapes" / "Ethiopia Electricity Transmission Network.shp"
TRANSMISSION_ADDITIONS_PATH = DATA_DIR / "transmission_additions.csv"
# Written by osm_grid.py: every OpenStreetMap power line in Ethiopia, parsed.
OSM_LINES_PATH = DATA_DIR / "shapes" / "osm_power_lines.gpkg"


def resolve_grid_source(source: Optional[str] = None) -> str:
    """'osm' or 'wb2007'. 'auto' (the default) prefers OSM when it has been
    downloaded: it is current and surveyed, whereas the 2006-07 routes were
    digitised from map archives and predate GERD and the railway."""
    if source in ("osm", "wb2007"):
        return source
    return "osm" if OSM_LINES_PATH.exists() else "wb2007"


def _osm_as_2007_schema():
    """OSM lines in the World Bank column layout, one row per circuit, so the
    same filtering and aggregation apply to both sources."""
    import geopandas as gpd
    osm = gpd.read_file(OSM_LINES_PATH).to_crs(4326)
    rows = []
    for _, r in osm.iterrows():
        for _ in range(max(int(r.get("circuits", 1) or 1), 1)):
            rows.append({"COUNTRY": "ETH", "VOLTAGE_KV": float(r.voltage_kv),
                         "FROM_NM": str(r.get("name") or f"osm {r.osm_id}"),
                         "TO_NM": "", "STATUS": "Existing",
                         "SOURCES": f"OpenStreetMap way {r.osm_id}",
                         "geometry": r.geometry})
    return gpd.GeoDataFrame(rows, crs=4326)

# Lines in these statuses count as built today. "Under construction" is
# included because this dataset reflects 2006-07 sources: anything under
# construction then (e.g. Mekelle-Tekeze 230 kV, Tekeze commissioned 2009)
# has almost certainly been completed since.
BUILT_STATUSES = ("existing", "under construction")

# ASSUMPTION -- MW per circuit by voltage class. The dataset gives voltage but
# no thermal rating or circuit count, as almost all public line data does.
# These are mid-range thermal ratings for common single-circuit overhead
# lines. Every line is assumed single-circuit; a double-circuit line would be
# understated by half. Edit these if you have better information.
CIRCUIT_MW = {66: 50.0, 132: 125.0, 230: 350.0, 400: 1100.0, 500: 1500.0}

# Security margin applied as PyPSA's s_max_pu, matching lines.s_max_pu in the
# PyPSA-Earth config. Approximates N-1 operation on a linearised network.
S_MAX_PU = 0.7

X_OHM_PER_KM = 0.40      # typical HV overhead line reactance
R_OVER_X = 0.10
S_BASE_MVA = 100.0
LENGTH_FACTOR = 1.25     # route length vs straight-line, as in PyPSA-Earth
ENDPOINT_SNAP_M = 25_000 # endpoints further than this outside Ethiopia are external
ETHIOPIA_UTM = 32637     # UTM 37N, metric CRS for lengths and distances


def _voltage_class(kv: float) -> int:
    return min(CIRCUIT_MW, key=lambda c: abs(c - kv))


def load_transmission_lines(
    path: Optional[Path] = None,
    statuses=BUILT_STATUSES,
    min_kv: float = 66.0,
    source: Optional[str] = None,
):
    """
    Returns (lines, excluded) GeoDataFrames, or (None, None) if absent.
    `excluded` keeps every dropped line with a reason, so nothing vanishes
    silently.
    """
    import geopandas as gpd

    if path is None and resolve_grid_source(source) == "osm":
        gdf = _osm_as_2007_schema()
    else:
        path = path or TRANSMISSION_PATH
        if not path.exists():
            return None, None
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)

    gdf["VOLTAGE_KV"] = pd.to_numeric(gdf["VOLTAGE_KV"], errors="coerce")
    status = gdf["STATUS"].fillna("").str.strip().str.lower()
    wanted = {s.lower() for s in statuses}

    reason = pd.Series("", index=gdf.index, dtype=object)
    reason[gdf.geometry.isna() | gdf.geometry.is_empty] = "no geometry"
    reason[(reason == "") & gdf["VOLTAGE_KV"].isna()] = "no voltage"
    reason[(reason == "") & (gdf["VOLTAGE_KV"] < min_kv)] = f"below {min_kv:g} kV"
    reason[(reason == "") & ~status.isin(wanted)] = (
        "status: " + gdf["STATUS"].fillna("blank").astype(str)
    )

    keep = reason == ""
    excluded = gdf[~keep].assign(reason=reason[~keep])
    return gdf[keep].copy(), excluded


def _line_endpoints(geom):
    """First and last point of a line. For a MultiLineString that will not
    merge into one piece, the two part-endpoints furthest apart."""
    from shapely.geometry import Point
    from shapely.ops import linemerge

    if geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        if merged.geom_type == "LineString":
            geom = merged
        else:
            pts = []
            for part in merged.geoms:
                pts.extend([Point(part.coords[0]), Point(part.coords[-1])])
            best, pair = -1.0, (pts[0], pts[-1])
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = pts[i].distance(pts[j])
                    if d > best:
                        best, pair = d, (pts[i], pts[j])
            return pair
    return Point(geom.coords[0]), Point(geom.coords[-1])


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return float(6371.0 * 2 * np.arcsin(np.sqrt(a)))


def load_transmission_additions(path: Optional[Path] = None,
                                source: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Hand-curated lines built after the base dataset. bus0,bus1 must be
    model bus names. Each row needs a source."""
    path = path or TRANSMISSION_ADDITIONS_PATH
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    if df.empty:
        return None
    df["voltage_kv"] = pd.to_numeric(df["voltage_kv"], errors="coerce")
    df["circuits"] = pd.to_numeric(df.get("circuits", 1), errors="coerce").fillna(1)
    # `base` says which network a row patches. Rows written to fill gaps in
    # the 2006-07 data (GERD's lines, OSM candidates) must NOT be applied on
    # top of OSM, which already contains them. Missing column = wb2007.
    base = df.get("base", pd.Series("wb2007", index=df.index)).fillna("wb2007").astype(str)
    df = df[base.isin([resolve_grid_source(source), "any"])]
    if df.empty:
        return None
    return df.dropna(subset=["bus0", "bus1", "voltage_kv"])


def aggregate_corridors(buses: pd.DataFrame, model_kv: float = 400.0,
                       source: Optional[str] = None):
    """
    Collapse real transmission lines into corridors between model buses.

    Returns (corridors, report). corridors has one row per connected bus pair
    with s_nom (MW, before S_MAX_PU), x and r in ohms referred to model_kv,
    and the lines it was built from. Returns (None, report) if the line data
    or region shapes are unavailable.

    Impedances are combined properly: each line is converted to per-unit on
    its OWN voltage base, the parallel combination is taken in per-unit, and
    the result is referred back to the model's bus voltage. Adding raw ohms
    across voltage levels would make a 66 kV line look as strong as a 230 kV
    one in the power flow.
    """
    import geopandas as gpd

    report: dict = {}
    source = resolve_grid_source(source)
    report["source"] = source
    lines, excluded = load_transmission_lines(source=source)
    if lines is None:
        report["status"] = "no transmission file"
        return None, report

    report["lines_total"] = len(lines) + len(excluded)
    report["excluded"] = excluded.drop(columns="geometry").to_dict("records") if len(excluded) else []
    report["excluded_by_reason"] = (
        excluded["reason"].value_counts().to_dict() if len(excluded) else {}
    )

    regions = bus_regions(buses).to_crs(ETHIOPIA_UTM)
    regions = regions.reset_index()[["bus", "geometry"]]

    rows = []
    for idx, line in lines.iterrows():
        a, b = _line_endpoints(line.geometry)
        rows.append({"line": idx, "end": 0, "geometry": a})
        rows.append({"line": idx, "end": 1, "geometry": b})
    ends = gpd.GeoDataFrame(rows, crs=4326).to_crs(ETHIOPIA_UTM)

    snapped = gpd.sjoin_nearest(
        ends, regions, how="left", max_distance=ENDPOINT_SNAP_M, distance_col="snap_m"
    )
    snapped = snapped[~snapped.index.duplicated(keep="first")]
    end_bus = snapped.pivot(index="line", columns="end", values="bus")

    lengths_km = lines.to_crs(ETHIOPIA_UTM).geometry.length / 1000.0

    internal, external, crossing = [], [], []
    for idx, line in lines.iterrows():
        b0, b1 = end_bus.loc[idx, 0], end_bus.loc[idx, 1]
        label = f"{line.get('FROM_NM', '?')}-{line.get('TO_NM', '?')} {line.VOLTAGE_KV:g}kV"
        if pd.isna(b0) or pd.isna(b1):
            external.append(label)
            continue
        if b0 == b1:
            internal.append(label)
            continue
        kv = float(line.VOLTAGE_KV)
        vclass = _voltage_class(kv)
        length = float(lengths_km.loc[idx])
        x_ohm = X_OHM_PER_KM * length
        crossing.append({
            "bus0": min(b0, b1), "bus1": max(b0, b1),
            "mw": CIRCUIT_MW[vclass],
            "x_pu": x_ohm / (kv ** 2 / S_BASE_MVA),
            "kv": kv, "label": label, "origin": "base dataset",
        })

    additions = load_transmission_additions(source=source)
    report["additions"] = 0
    if additions is not None:
        coords = buses[["x", "y"]]
        for _, add in additions.iterrows():
            if add.bus0 not in coords.index or add.bus1 not in coords.index:
                continue
            kv = float(add.voltage_kv)
            dist = _haversine_km(coords.loc[add.bus0, "x"], coords.loc[add.bus0, "y"],
                                 coords.loc[add.bus1, "x"], coords.loc[add.bus1, "y"])
            length = dist * LENGTH_FACTOR
            for _ in range(int(add.circuits)):
                crossing.append({
                    "bus0": min(add.bus0, add.bus1), "bus1": max(add.bus0, add.bus1),
                    "mw": CIRCUIT_MW[_voltage_class(kv)],
                    "x_pu": X_OHM_PER_KM * length / (kv ** 2 / S_BASE_MVA),
                    "kv": kv, "label": f"{add.get('name', 'addition')} {kv:g}kV",
                    "origin": "addition",
                })
                report["additions"] += 1

    report["lines_used"] = len(lines)
    report["internal"] = internal
    report["external"] = external
    report["crossing"] = len([c for c in crossing if c["origin"] == "base dataset"])

    if not crossing:
        report["status"] = "no lines cross between model regions"
        return None, report

    df = pd.DataFrame(crossing)
    z_base_model = model_kv ** 2 / S_BASE_MVA
    out = []
    for (b0, b1), grp in df.groupby(["bus0", "bus1"]):
        x_pu = 1.0 / (1.0 / grp.x_pu).sum()
        dist = _haversine_km(buses.loc[b0, "x"], buses.loc[b0, "y"],
                             buses.loc[b1, "x"], buses.loc[b1, "y"])
        out.append({
            "bus0": b0, "bus1": b1,
            "s_nom": float(grp.mw.sum()),
            "x": x_pu * z_base_model,
            "r": x_pu * z_base_model * R_OVER_X,
            "length": dist * LENGTH_FACTOR,
            "n_lines": len(grp),
            "voltages": ", ".join(f"{v:g}" for v in sorted(grp.kv.unique())),
            "lines": "; ".join(grp.label),
            "origin": "+".join(sorted(grp.origin.unique())),
        })
    corridors = pd.DataFrame(out)

    connected = set(corridors.bus0) | set(corridors.bus1)
    report["isolated_buses"] = sorted(set(buses.index) - connected)
    report["status"] = "ok"
    return corridors, report
