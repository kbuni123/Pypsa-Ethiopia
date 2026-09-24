"""
osm_grid.py -- find transmission lines built since the 2006-07 network data.

    python osm_grid.py                 # download (cached), compare, report
    python osm_grid.py --refresh       # force a fresh download
    python osm_grid.py --min-kv 132    # ignore sub-transmission

What it does
------------
1. Downloads every power line mapped in Ethiopia from OpenStreetMap
   (Overpass API), all voltages. Cached to data/shapes/osm_power_lines.json.
2. Compares each OSM line against the 2007 World Bank / EEPCo network. A line
   counts as ALREADY KNOWN only if most of its length runs alongside a 2007 line
   of the same or higher voltage -- so an upgrade on an old route (e.g. 132 ->
   230 kV) still shows up as new.
3. Aggregates the NEW lines into model corridors, exactly as the model does,
   and writes them to data/osm_transmission_candidates.csv.

NOTHING IS ADDED TO THE MODEL AUTOMATICALLY. Review the candidates, then copy
the rows you accept into data/transmission_additions.csv and run
`python setup_data.py --grid` to see the effect.

Caveats
-------
* OSM has no thermal ratings either. Capacity is still estimated from voltage
  (data.CIRCUIT_MW), with circuits read from the `circuits` or `cables` tags.
* OSM completeness in Ethiopia is uneven. A missing line in OSM is not proof
  the line does not exist.
* HVDC lines (frequency=0), such as the Sodo-Moyale link to Kenya, are
  excluded: they are point-to-point export links, not AC corridors, and
  exports are already modelled as load.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import data as D

OSM_JSON = D.DATA_DIR / "shapes" / "osm_power_lines.json"
OSM_GPKG = D.DATA_DIR / "shapes" / "osm_power_lines.gpkg"
CANDIDATES_CSV = D.DATA_DIR / "osm_transmission_candidates.csv"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

QUERY = """
[out:json][timeout:600];
area["ISO3166-1"="ET"]["admin_level"="2"]->.et;
(
  way["power"="line"](area.et);
  way["power"="cable"](area.et);
);
out geom;
"""


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download(refresh: bool = False) -> dict:
    if OSM_JSON.exists() and not refresh:
        print(f"  using cached download: {OSM_JSON}")
        return json.loads(OSM_JSON.read_text(encoding="utf-8"))

    body = urllib.parse.urlencode({"data": QUERY}).encode()
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        for attempt in range(2):
            try:
                print(f"  querying {url} (can take a few minutes)...")
                req = urllib.request.Request(url, data=body,
                                             headers={"User-Agent": "pypsa-ethiopia/1.0"})
                with urllib.request.urlopen(req, timeout=900) as resp:
                    raw = resp.read().decode("utf-8")
                result = json.loads(raw)
                OSM_JSON.parent.mkdir(parents=True, exist_ok=True)
                OSM_JSON.write_text(raw, encoding="utf-8")
                print(f"  saved {len(result.get('elements', []))} elements to {OSM_JSON}")
                return result
            except Exception as exc:
                last_err = exc
                print(f"    failed: {type(exc).__name__}: {exc}")
                time.sleep(10)
    sys.exit(f"All Overpass endpoints failed ({last_err}). Try again later -- "
             "the public servers are sometimes busy.")


# --------------------------------------------------------------------------
# Parsing OSM tags
# --------------------------------------------------------------------------
def parse_voltages_kv(tag: str | None) -> list[float]:
    """'400000;230000' -> [400.0, 230.0]. Values under 1000 are taken as kV."""
    if not tag:
        return []
    out = []
    for part in str(tag).replace(",", ";").split(";"):
        digits = "".join(ch for ch in part if ch.isdigit() or ch == ".")
        if not digits:
            continue
        v = float(digits)
        out.append(v / 1000.0 if v >= 1000 else v)
    return out


def parse_circuits(tags: dict, n_voltages: int) -> list[int]:
    """Circuits per voltage. Uses `circuits`, else `cables` / 3, else 1."""
    if tags.get("circuits"):
        vals = [p for p in str(tags["circuits"]).split(";") if p.strip().isdigit()]
        if vals:
            vals = [int(v) for v in vals]
            if len(vals) == n_voltages:
                return vals
            return [max(1, vals[0] // max(n_voltages, 1))] * n_voltages
    if tags.get("cables"):
        vals = [p for p in str(tags["cables"]).split(";") if p.strip().isdigit()]
        if vals:
            total = sum(int(v) for v in vals)
            per = max(1, total // 3 // max(n_voltages, 1))
            return [per] * n_voltages
    return [1] * n_voltages


def to_frame(result: dict):
    """One row per (way, voltage), carrying its circuit count."""
    import geopandas as gpd
    from shapely.geometry import LineString

    rows, stats = [], {"ways": 0, "no_voltage": 0, "dc": 0, "no_geometry": 0}
    for el in result.get("elements", []):
        if el.get("type") != "way":
            continue
        stats["ways"] += 1
        tags = el.get("tags", {})
        geom = el.get("geometry") or []
        if len(geom) < 2:
            stats["no_geometry"] += 1
            continue
        if str(tags.get("frequency", "")).strip() == "0":
            stats["dc"] += 1
            continue
        volts = parse_voltages_kv(tags.get("voltage"))
        if not volts:
            stats["no_voltage"] += 1
            continue
        line = LineString([(p["lon"], p["lat"]) for p in geom])
        for kv, circ in zip(volts, parse_circuits(tags, len(volts))):
            rows.append({
                "osm_id": el["id"], "voltage_kv": kv, "circuits": circ,
                "name": tags.get("name") or tags.get("ref") or "",
                "operator": tags.get("operator", ""),
                "geometry": line,
            })
    gdf = gpd.GeoDataFrame(rows, crs=4326) if rows else gpd.GeoDataFrame(
        columns=["osm_id", "voltage_kv", "circuits", "name", "operator", "geometry"],
        geometry="geometry", crs=4326)
    return gdf, stats


# --------------------------------------------------------------------------
# Comparison with the 2007 network
# --------------------------------------------------------------------------
def mark_known(osm, old, buffer_km: float, overlap: float):
    """
    `known` = at least `overlap` of the OSM line's length lies within
    `buffer_km` of a 2007 line whose voltage class is >= its own.
    """
    osm_m = osm.to_crs(D.ETHIOPIA_UTM)
    old_m = old.to_crs(D.ETHIOPIA_UTM)
    old_m = old_m.assign(vclass=old_m["VOLTAGE_KV"].map(D._voltage_class))

    buffers = {}
    for vc in sorted(D.CIRCUIT_MW):
        eligible = old_m[old_m.vclass >= vc]
        buffers[vc] = eligible.buffer(buffer_km * 1000).union_all() if len(eligible) else None

    frac = []
    for _, row in osm_m.iterrows():
        buf = buffers.get(D._voltage_class(row.voltage_kv))
        length = row.geometry.length
        if buf is None or length <= 0:
            frac.append(0.0)
        else:
            frac.append(row.geometry.intersection(buf).length / length)
    osm = osm.copy()
    osm["overlap_2007"] = frac
    osm["known"] = osm["overlap_2007"] >= overlap
    osm["length_km"] = osm_m.geometry.length / 1000.0
    return osm


def assign_buses(gdf, regions_m):
    """Endpoint -> model bus, same rule as data.aggregate_corridors."""
    import geopandas as gpd

    rows = []
    for idx, line in gdf.iterrows():
        a, b = D._line_endpoints(line.geometry)
        rows += [{"row": idx, "end": 0, "geometry": a}, {"row": idx, "end": 1, "geometry": b}]
    ends = gpd.GeoDataFrame(rows, crs=4326).to_crs(D.ETHIOPIA_UTM)
    snapped = gpd.sjoin_nearest(ends, regions_m, how="left",
                                max_distance=D.ENDPOINT_SNAP_M, distance_col="snap_m")
    snapped = snapped[~snapped.index.duplicated(keep="first")]
    ends_bus = snapped.pivot(index="row", columns="end", values="bus")
    out = gdf.copy()
    out["bus0"] = ends_bus[0].reindex(out.index)
    out["bus1"] = ends_bus[1].reindex(out.index)
    return out


def corridors_from(gdf, buses):
    """Model corridors for a line set, via the model's own aggregation."""
    if gdf is None or len(gdf) == 0:
        return pd.DataFrame(columns=["bus0", "bus1", "s_nom"])
    rows = []
    for _, r in gdf.iterrows():
        for _ in range(int(r.circuits)):
            rows.append({"COUNTRY": "ETH", "VOLTAGE_KV": r.voltage_kv,
                         "FROM_NM": str(r.osm_id), "TO_NM": r.get("name", ""),
                         "STATUS": "Existing", "geometry": r.geometry})
    import geopandas as gpd
    frame = gpd.GeoDataFrame(rows, crs=4326)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lines.gpkg"
        frame.to_file(path, driver="GPKG")
        saved = (D.TRANSMISSION_PATH, D.TRANSMISSION_ADDITIONS_PATH)
        try:
            D.TRANSMISSION_PATH = path
            D.TRANSMISSION_ADDITIONS_PATH = Path(tmp) / "none.csv"
            corr, _ = D.aggregate_corridors(buses, source="wb2007")
        finally:
            D.TRANSMISSION_PATH, D.TRANSMISSION_ADDITIONS_PATH = saved
    return corr if corr is not None else pd.DataFrame(columns=["bus0", "bus1", "s_nom"])


def corridors_2007(buses):
    saved = D.TRANSMISSION_ADDITIONS_PATH
    try:
        D.TRANSMISSION_ADDITIONS_PATH = Path(tempfile.gettempdir()) / "no_additions.csv"
        corr, _ = D.aggregate_corridors(buses, source="wb2007")
    finally:
        D.TRANSMISSION_ADDITIONS_PATH = saved
    return corr if corr is not None else pd.DataFrame(columns=["bus0", "bus1", "s_nom"])


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="re-download from Overpass")
    p.add_argument("--min-kv", type=float, default=66.0)
    p.add_argument("--buffer-km", type=float, default=5.0,
                   help="how close an OSM line must run to a 2007 line to count as the same")
    p.add_argument("--overlap", type=float, default=0.7,
                   help="share of an OSM line's length that must be near a 2007 line")
    args = p.parse_args()

    from model import REGIONS
    buses = pd.DataFrame({"x": [r.lon for r in REGIONS], "y": [r.lat for r in REGIONS]},
                         index=[r.name for r in REGIONS])
    for path, what in ((D.GADM_PATH, "GADM shapes (python setup_data.py --shapes)"),
                       (D.TRANSMISSION_PATH, "the 2007 transmission shapefile")):
        if not path.exists():
            sys.exit(f"Missing {what}: {path}")

    print("[1/4] OpenStreetMap power lines")
    result = download(args.refresh)
    osm_all, stats = to_frame(result)
    print(f"  {stats['ways']} ways | skipped: {stats['dc']} HVDC, "
          f"{stats['no_voltage']} with no voltage tag, {stats['no_geometry']} without geometry")
    if len(osm_all):
        counts = osm_all.groupby(osm_all.voltage_kv.round(0)).agg(
            ways=("osm_id", "nunique"), circuits=("circuits", "sum"))
        counts["km"] = osm_all.to_crs(D.ETHIOPIA_UTM).groupby(
            osm_all.voltage_kv.round(0)).geometry.apply(lambda g: g.length.sum() / 1000).round(0)
        print("  mapped lines by voltage (kV):")
        print("    " + counts.to_string().replace("\n", "\n    "))
        osm_all.to_file(OSM_GPKG, driver="GPKG")

    osm = osm_all[osm_all.voltage_kv >= args.min_kv].copy()
    if osm.empty:
        sys.exit(f"No mapped lines at or above {args.min_kv:g} kV with a voltage tag.")

    print(f"\n[2/4] comparing {len(osm)} line-voltages >= {args.min_kv:g} kV with the 2007 network")
    old, _ = D.load_transmission_lines(source="wb2007")
    osm = mark_known(osm, old, args.buffer_km, args.overlap)
    print(f"  {int(osm.known.sum())} follow a 2007 line of equal or higher voltage")
    print(f"  {int((~osm.known).sum())} are NEW or UPGRADED since the 2007 data")

    print("\n[3/4] corridors: 2007 vs OSM (MW nameplate, before the 70% derate)")
    c07 = corridors_2007(buses).set_index(["bus0", "bus1"])["s_nom"].rename("2007")
    cosm = corridors_from(osm, buses).set_index(["bus0", "bus1"])["s_nom"].rename("OSM, all")
    cnew = corridors_from(osm[~osm.known], buses).set_index(["bus0", "bus1"])["s_nom"].rename("OSM, new only")
    table = pd.concat([c07, cosm, cnew], axis=1).fillna(0).round(0).astype(int)
    table.index = [f"{a} - {b}" for a, b in table.index]
    print("  " + table.sort_index().to_string().replace("\n", "\n  "))

    print("\n[4/4] candidate additions (new/upgraded lines that cross between model regions)")
    regions_m = D.bus_regions(buses).to_crs(D.ETHIOPIA_UTM).reset_index()[["bus", "geometry"]]
    new = assign_buses(osm[~osm.known], regions_m)
    cross = new[new.bus0.notna() & new.bus1.notna() & (new.bus0 != new.bus1)].copy()
    if cross.empty:
        print("  none -- no new line crosses between model regions")
        return
    cross["pair"] = [tuple(sorted((a, b))) for a, b in zip(cross.bus0, cross.bus1)]

    existing = D.load_transmission_additions(source="wb2007")
    already = set()
    if existing is not None:
        for _, r in existing.iterrows():
            already.add((tuple(sorted((r.bus0, r.bus1))), D._voltage_class(float(r.voltage_kv))))

    out, skipped = [], []
    for (pair, kv), grp in cross.groupby(["pair", cross.voltage_kv.round(0)]):
        names = sorted({n for n in grp.name if n})
        km = float(grp.length_km.sum())
        if (pair, D._voltage_class(kv)) in already:
            skipped.append(f"{pair[0]}-{pair[1]} {kv:g} kV ({int(grp.circuits.sum())} circuits)")
            continue
        # Same seven columns as transmission_additions.csv, so rows paste
        # straight in. Extras live in `source`; '#' is stripped because the
        # additions loader treats it as a comment marker.
        src = (f"OSM ways {' '.join(str(i) for i in sorted(grp.osm_id.unique()))}; "
               f"{km:.0f} km crossing; " + (f"names: {'; '.join(names)}" if names else "unnamed"))
        out.append({
            "name": f"OSM {kv:g} kV {pair[0]}-{pair[1]}",
            "bus0": pair[0], "bus1": pair[1], "voltage_kv": kv,
            "circuits": int(grp.circuits.sum()), "status": "osm-candidate", "base": "wb2007",
            "source": src.replace("#", "")[:300],
        })
    if skipped:
        print("  already in transmission_additions.csv, left out: " + ", ".join(skipped))
    if not out:
        print("  no further candidates")
        return
    # Column ORDER must match transmission_additions.csv exactly, since rows
    # are pasted in positionally.
    cand = pd.DataFrame(out).sort_values(["bus0", "bus1", "voltage_kv"])[
        ["name", "bus0", "bus1", "voltage_kv", "circuits", "status", "source", "base"]]
    cand.to_csv(CANDIDATES_CSV, index=False)

    show = cand[["bus0", "bus1", "voltage_kv", "circuits"]].copy()
    show["est_MW"] = [D.CIRCUIT_MW[D._voltage_class(v)] * c for v, c in zip(cand.voltage_kv, cand.circuits)]
    print("  " + show.to_string(index=False).replace("\n", "\n  "))
    print(f"\n  written to {CANDIDATES_CSV}")
    print("  Same columns as data/transmission_additions.csv: copy the rows you")
    print("  accept (not the header) to the end of that file, then run:")
    print("      python setup_data.py --grid")


if __name__ == "__main__":
    main()
