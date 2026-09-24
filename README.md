# PyPSA-Ethiopia

A six-node capacity-expansion and dispatch model of the Ethiopian power system,
driven from Streamlit. No Snakemake, no Zenodo data bundle, no GDAL.

## Quick start

```bash
pip install -r requirements.txt
python model.py          # headless self-test, should print non-zero KPIs
streamlit run app.py
```

Or with Docker:

```bash
docker compose up --build   # then open http://localhost:8501
```

## What was wrong before

The previous version built its time series like this:

```python
n = pypsa.Network()                                    # no snapshots yet
n.add("Load", "l", bus=b,
      p_set=np.random.uniform(100, 500, len(n.snapshots)))   # length 0
n.add("Generator", "g", bus=b,
      p_max_pu=np.random.uniform(0, 1, len(n.snapshots)))    # length 0
n.set_snapshots(pd.date_range(...))                    # far too late
```

A fresh `pypsa.Network()` carries no user snapshots, so `len(n.snapshots)` was
0 and both arrays came out empty. Every load ended up with zero demand and
every generator with zero availability. The LP was then trivially satisfied by
building nothing, which is why the app reported **€0 total cost, 0 MWh
generation and 0.0 % renewable share**. PyPSA and HiGHS were fine the whole
time — the model was empty.

`model.build_network()` now sets snapshots as its first statement and passes
the snapshot index explicitly into every profile function, so the mistake
cannot be repeated silently. `model.validate_network()` re-checks it anyway and
the Run tab refuses to call the solver if any check fails.

## Layout

| File | Purpose |
|---|---|
| `model.py` | Regions, costs, profiles, network builder, validation, KPIs. No Streamlit import, so it is testable headless. |
| `app.py` | Streamlit UI: Run, Results, Dispatch, Network, Inputs, Diagnostics, Export. |
| `Dockerfile` | Python 3.11 slim, adds `curl` so the healthcheck actually works. |
| `docker-compose.yml` | Single service; the unused Postgres and Redis services are gone. |

## Model

- **Nodes**: Addis Ababa, Bahir Dar, Mekelle, Hawassa, Dire Dawa, Jigjiga,
  linked by six 400 kV corridors with optional expansion.
- **Demand**: national peak split by regional share, evening-peaking diurnal
  shape, weekend dip, mild seasonal swing.
- **Solar**: clear-sky geometry from latitude and day of year, scaled by
  regional clearness and damped through the Kiremt rains.
- **Wind**: AR(1) series with an afternoon diurnal peak and a Bega-season
  uplift, rescaled to each region's mean capacity factor.
- **Hydro**: existing fleet at fixed capacity plus optional new build, both on
  a seasonal inflow profile.
- **Also available**: geothermal in the Rift Valley nodes, diesel backup,
  4-hour batteries at every node.
- **Slack**: a load-shedding generator at €3000/MWh sits at every bus, so the
  problem is always feasible and a capacity shortfall shows up as unserved
  energy instead of an infeasible solve.

Profiles are deterministic — the same scenario always reproduces the same
numbers. Annualised CAPEX is scaled by the modelled fraction of a year so the
objective stays comparable across horizon lengths.

## Swapping in real data

The profile functions are the only place synthetic data enters. Replace them
with your own series and nothing else changes:

```python
def solar_profile(sn, region):
    cf = pd.read_csv(f"data/solar_{region.name}.csv",
                     index_col=0, parse_dates=True).squeeze()
    return cf.reindex(sn).interpolate().clip(0, 1)   # must match sn exactly
```

The one rule: whatever you return must be indexed by the snapshot index `sn`.
That is the whole lesson from the bug above.

## Diagnostics tab

- **Reproduce the old bug** — shows the empty-array behaviour in situ.
- **Run smoke test** — a 24-hour one-bus model that proves PyPSA and the solver
  are working before you blame the big model.
- Component tables and time-series sums, which should never be zero.
