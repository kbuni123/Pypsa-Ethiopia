"""
setup_data.py -- one-time download and preprocessing for PyPSA-Ethiopia.

    python setup_data.py --all          # everything, in order
    python setup_data.py --shapes       # GADM admin boundaries (small, fast)
    python setup_data.py --costs        # technology-data costs (small, fast)
    python setup_data.py --cutout       # ERA5 download via CDS (SLOW, GBs)
    python setup_data.py --profiles     # atlite -> cached capacity factors
    python setup_data.py --grid         # check the transmission aggregation

Order matters: shapes and cutout must exist before profiles.

CDS SETUP (needed for --cutout only)
------------------------------------
1. Register at https://cds.climate.copernicus.eu and accept the ERA5 licence.
2. Put your key in ~/.cdsapirc:

       url: https://cds.climate.copernicus.eu/api
       key: <your-CDS-API-key>

3. Expect the ERA5 request to queue. A one-year Ethiopia cutout at 0.3 deg is
   roughly 2-4 GB and can sit in the CDS queue for hours. Start it and go do
   something else. This is the price of the "most correct" path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import data as D

# Bus coordinates must match model.REGIONS.
from model import REGIONS

TECHNOLOGY_DATA_URL = (
    "https://raw.githubusercontent.com/PyPSA/technology-data/"
    "master/outputs/costs_2030.csv"
)
# GADM 4.1 mirror used by PyPSA-Earth. If this 404s, download the Ethiopia
# GeoPackage manually from https://gadm.org/download_country.html and save it
# to data/shapes/gadm41_ETH.gpkg
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_ETH.gpkg"

# Ethiopia bounding box, with a small margin
X_RANGE = (32.5, 48.5)
Y_RANGE = (3.0, 15.2)
WEATHER_YEAR = 2013


def buses_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"x": [r.lon for r in REGIONS], "y": [r.lat for r in REGIONS]},
        index=[r.name for r in REGIONS],
    )


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    print(f"          -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  done ({dest.stat().st_size / 1e6:.1f} MB)")


def fetch_shapes() -> None:
    print("[shapes] GADM level-1 boundaries for Ethiopia")
    if D.GADM_PATH.exists():
        print(f"  already present: {D.GADM_PATH}")
        return
    try:
        _download(GADM_URL, D.GADM_PATH)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        print("  Download gadm41_ETH.gpkg manually from")
        print("    https://gadm.org/download_country.html")
        print(f"  and save it to {D.GADM_PATH}")
        raise


def fetch_costs() -> None:
    print("[costs] technology-data costs_2030.csv")
    if D.COSTS_PATH.exists():
        print(f"  already present: {D.COSTS_PATH}")
        return
    _download(TECHNOLOGY_DATA_URL, D.COSTS_PATH)

    found = D.real_tech_costs()
    if found:
        print(f"  parsed {len(found)} technologies: {', '.join(sorted(found))}")
        for carrier, entry in sorted(found.items()):
            capex = entry.get("capex")
            if capex:
                print(f"    {carrier:11s} capex {capex:>12,.0f} EUR/MW")
    else:
        print("  WARNING: file downloaded but no technologies matched.")
        print("  Check data.COST_TECH_MAP against the technology names in the CSV.")


# "height" is ERA5 orography. convert_runoff weights runoff by it, so the hydro
# profile fails without it. It is a single static field -- a tiny download.
CUTOUT_FEATURES = ["height", "influx", "temperature", "wind", "runoff"]


def _cutout_vars(path: Path) -> set:
    """Variable names in a netCDF file, with the handle closed on return."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return set(ds.data_vars)


def recover_tmp_cutout() -> bool:
    """
    Finish a cutout update that atlite wrote but could not swap into place.

    atlite.prepare() writes the merged cutout to a temp file beside the
    original, deletes the original, then renames the temp file over it. On
    Windows the delete can fail with WinError 32 because a lazy xarray handle
    still holds the original open -- leaving a complete temp file stranded
    next to an untouched original. In a fresh process nothing holds either
    file, so the swap can be finished safely here.

    Only swaps in a temp file that opens cleanly, holds every variable the
    original has, and includes 'height'. Anything else is left alone.
    """
    import gc
    import os

    directory = D.CUTOUT_PATH.parent
    if not directory.exists():
        return False
    candidates = sorted(
        directory.glob(f"tmp*{D.CUTOUT_PATH.name}"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        return False

    print(f"  found {len(candidates)} temp cutout(s) from an unfinished update")
    original_vars = _cutout_vars(D.CUTOUT_PATH) if D.CUTOUT_PATH.exists() else set()

    for tmp in candidates:
        try:
            tmp_vars = _cutout_vars(tmp)
        except Exception as exc:
            print(f"  skipping {tmp.name}: unreadable ({type(exc).__name__})")
            continue
        if not original_vars <= tmp_vars or "height" not in tmp_vars:
            print(f"  skipping {tmp.name}: does not contain the complete feature set")
            continue

        gc.collect()
        os.replace(tmp, D.CUTOUT_PATH)
        print(f"  recovered: {tmp.name} -> {D.CUTOUT_PATH.name}")

        for leftover in candidates:
            if leftover != tmp and leftover.exists():
                leftover.unlink()
                print(f"  removed stale temp file {leftover.name}")
        return True

    print("  no usable temp cutout -- leaving files as they are")
    return False


def build_cutout(year: int = WEATHER_YEAR) -> None:
    print(f"[cutout] ERA5 for Ethiopia, {year}")

    try:
        import atlite
    except ImportError:
        sys.exit("atlite is not installed. pip install atlite cdsapi")

    if not (Path.home() / ".cdsapirc").exists():
        sys.exit(
            "No ~/.cdsapirc found. See the CDS SETUP notes at the top of this file."
        )

    # Finish any swap a previous run left half-done (Windows file locking).
    # Must run BEFORE atlite opens the cutout, or this process holds the lock.
    recover_tmp_cutout()

    if D.CUTOUT_PATH.exists():
        # atlite only fetches features the file does not already hold, so
        # re-preparing an existing cutout tops it up rather than redownloading.
        existing = atlite.Cutout(str(D.CUTOUT_PATH))
        missing = [f for f in CUTOUT_FEATURES
                   if f not in existing.prepared_features.index.get_level_values("feature")]
        if not missing:
            print(f"  already complete: {D.CUTOUT_PATH}")
            return
        print(f"  existing cutout is missing: {', '.join(missing)} -- fetching only those")
        try:
            existing.prepare(features=missing)
        except PermissionError:
            # Windows: atlite wrote the merged file but could not delete the
            # original it still had open. Drop our handles and finish the swap.
            import gc
            try:
                existing.data.close()
            except Exception:
                pass
            del existing
            gc.collect()
            print("  Windows held the original file open -- finishing the swap")
            try:
                if recover_tmp_cutout():
                    print(f"  done: {D.CUTOUT_PATH}")
                    return
            except PermissionError:
                pass
            print("  The download succeeded but the file is still locked.")
            print("  Run  python setup_data.py --cutout  again -- a fresh process")
            print("  will finish the swap without downloading anything.")
            return
        print(f"  done: {D.CUTOUT_PATH} ({D.CUTOUT_PATH.stat().st_size / 1e9:.2f} GB)")
        return

    D.CUTOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cutout = atlite.Cutout(
        path=str(D.CUTOUT_PATH),
        module="era5",
        x=slice(*X_RANGE),
        y=slice(*Y_RANGE),
        time=str(year),
        dx=0.3,
        dy=0.3,
    )
    print("  requesting from CDS -- this queues and can take hours")
    cutout.prepare(features=CUTOUT_FEATURES)
    print(f"  done: {D.CUTOUT_PATH} ({D.CUTOUT_PATH.stat().st_size / 1e9:.2f} GB)")


def build_profiles() -> None:
    print("[profiles] area-weighted capacity factors per bus")
    if not D.cutout_available():
        sys.exit(f"No cutout at {D.CUTOUT_PATH}. Run --cutout first.")
    if not D.GADM_PATH.exists():
        sys.exit(f"No shapes at {D.GADM_PATH}. Run --shapes first.")

    buses = buses_frame()
    print(f"  aggregating over {len(buses)} bus regions")
    profiles, failures = D.compute_profiles_from_cutout(buses)

    for kind, df in profiles.items():
        means = df.mean()
        print(f"  {kind}: mean CF per bus  (saved)")
        for bus, cf in means.items():
            print(f"    {bus:14s} {cf:.3f}")

    print(f"  cached {len(profiles)} of 3 profiles under {D.PROFILE_DIR}")
    for kind, err in failures.items():
        print(f"  FAILED {kind}: {err}")
    if "hydro" in failures and "height" in failures["hydro"]:
        print("  -> the cutout lacks the 'height' feature. Run --cutout again;")
        print("     it will fetch only the missing feature, not the whole year.")

    print()
    print("  SANITY CHECK: solar CF should land around 0.20-0.26 for Ethiopia")
    print("  and wind well below solar except in Tigray and the Somali region.")
    print("  Numbers far outside that mean something went wrong upstream.")


def inspect_grid() -> None:
    """Aggregate the real transmission lines and show exactly what happened
    to every one of them. Nothing is written -- this is for checking."""
    print("[grid] transmission lines -> model corridors")
    if not D.TRANSMISSION_PATH.exists():
        sys.exit(f"No shapefile at {D.TRANSMISSION_PATH}")
    if not D.GADM_PATH.exists():
        sys.exit(f"No shapes at {D.GADM_PATH}. Run --shapes first.")

    corridors, rep = D.aggregate_corridors(buses_frame())
    print(f"  source: {rep.get('source')}  (osm when osm_grid.py has run; the app can switch)")
    print(f"  {rep.get('lines_total', 0)} lines in the file")
    for reason, count in rep.get("excluded_by_reason", {}).items():
        print(f"    excluded {count:3d}  {reason}")
    print(f"    {len(rep.get('internal', [])):3d} internal to one region (invisible at 6 nodes)")
    print(f"    {len(rep.get('external', [])):3d} with an end outside Ethiopia (dropped)")
    for label in rep.get("external", []):
        print(f"             {label}")
    print(f"    {rep.get('crossing', 0):3d} cross between regions -> corridors")
    if rep.get("additions"):
        print(f"    {rep['additions']:3d} circuits from transmission_additions.csv")

    if corridors is None:
        print(f"  no corridors built: {rep.get('status')}")
        return

    print()
    pd.set_option("display.width", 200)
    show = corridors[["bus0", "bus1", "s_nom", "n_lines", "voltages", "length"]].copy()
    show["usable_mw"] = show.s_nom * D.S_MAX_PU
    print(show.round(0).to_string(index=False))

    iso = rep.get("isolated_buses", [])
    if iso:
        print()
        print(f"  ISOLATED in the real data: {', '.join(iso)}")
        print("  The model will patch these with the synthetic corridor and flag it.")
        print("  If a real line should connect them, add it to transmission_additions.csv.")

    print()
    print("  Capacities are ESTIMATES from voltage class, single circuit assumed:")
    print("    " + ", ".join(f"{k} kV = {v:.0f} MW" for k, v in D.CIRCUIT_MW.items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true")
    p.add_argument("--shapes", action="store_true")
    p.add_argument("--costs", action="store_true")
    p.add_argument("--cutout", action="store_true")
    p.add_argument("--profiles", action="store_true")
    p.add_argument("--grid", action="store_true",
                   help="aggregate the transmission shapefile and report (writes nothing)")
    p.add_argument("--year", type=int, default=WEATHER_YEAR)
    args = p.parse_args()

    if not any([args.all, args.shapes, args.costs, args.cutout, args.profiles, args.grid]):
        p.print_help()
        return

    if args.all or args.shapes:
        fetch_shapes()
    if args.all or args.costs:
        fetch_costs()
    if args.all or args.cutout:
        build_cutout(args.year)
    if args.all or args.profiles:
        build_profiles()
    if args.grid:
        inspect_grid()

    print()
    print("Current data state:")
    print(D.provenance().to_string(index=False))


if __name__ == "__main__":
    main()
