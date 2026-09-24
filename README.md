# PyPSA-Ethiopia

A six-node capacity-expansion and dispatch model of Ethiopia's power system, built with [PyPSA](https://pypsa.org) and run from a Streamlit app. It is anchored to Ethiopian Electric Power's (EEP) reported FY2025/26 generation and uses ERA5 weather, OpenStreetMap transmission data and PyPSA technology-data costs.

The model was built to answer a current question: **how should Ethiopia's hydro-dominated grid handle demand growth and drought, and what role does curtailing crypto-mining play?**

## Headline results

Full results, sensitivities and caveats are in **[RESULTS.md](RESULTS.md)**.

- **Mining is the system's shock absorber.** In every scenario, the least-cost response to a shortfall is to curtail crypto miners before building capacity or cutting other customers.
- **The model brackets EEP's September 2026 decision.** In a 2026 drought with 20% less water, it serves between 4% and 34% of miners, depending on how much stored water is used. EEP's actual cut to 23% falls inside that range, and only for a water shortfall of about 20–25%, which matches the reported drought.
- **Growth plus drought requires new capacity by 2030.** With 5% annual demand growth and a dry year, the model builds about 1.3 GW of solar to protect non-mining customers, regardless of financing cost.
- **Solar to keep miners running is a policy choice.** It pays off only if mining is valued above about €30/MWh (8% discount rate) or €40/MWh (12%).

## The model

| Component | Treatment |
|---|---|
| Network | Six nodes (Addis Ababa, Bahir Dar, Mekelle, Hawassa, Dire Dawa, Jigjiga), with corridors aggregated from real transmission lines |
| Weather | ERA5 reanalysis via [atlite](https://github.com/PyPSA/atlite): area-weighted solar, wind and runoff profiles per node |
| Hydropower | Each large plant is a reservoir (PyPSA StorageUnit) with inflow scaled to its design energy. *Sustainable* mode follows an annual rule curve; *drawdown* mode lets GERD and Beles use stored water |
| Demand | Calibrated to EEP's FY2025/26 generation (35,671 GWh), split into existing customers, crypto mining (27% of domestic sales), Kenya and Djibouti exports, and new household connections |
| Costs | [PyPSA technology-data](https://github.com/PyPSA/technology-data) 2030 values, annualised at a chosen discount rate |
| Mining curtailment | Allowed at the mining tariff (€28.9/MWh), adjustable in the app |
| Unserved load | Priced at €3,000/MWh, so shortfalls show up as results instead of infeasible solves |
| Solver | HiGHS (bundled, no licence needed) |

Every assumption without a published source is marked `ASSUMPTION` in `data/*.yaml`, with the reasoning next to it.

## Quick start

Requires Python 3.11.

```bash
git clone https://github.com/kbuni123/Pypsa-Ethiopia.git
cd Pypsa-Ethiopia
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python model.py                 # headless self-test
streamlit run app.py            # open http://localhost:8501
```

Or with Docker, which mounts `data/` from your working copy:

```bash
docker compose up --build
```

## Data

The repository includes everything needed to run the model: the cached ERA5 capacity-factor profiles (`data/profiles/`), technology costs, plant and hydro data, and the demand and access assumptions.

Two data sets are **not included** because of their size, and `setup_data.py` rebuilds them:

| Data | Needed for | How to get it |
|---|---|---|
| ERA5 cutout (2–4 GB) | Regenerating weather profiles | Register at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu), put your key in `~/.cdsapirc`, then run `python setup_data.py --cutout --profiles`. The CDS queue can take hours |
| Region shapes and grid | The real transmission network | `python setup_data.py --shapes` for GADM boundaries; place the World Bank "Ethiopia Electricity Transmission Network" shapefile in `data/shapes/`; run `python osm_grid.py` to add lines mapped in OpenStreetMap |

Without the grid files, the app falls back to a simplified synthetic grid, so results will differ slightly from `RESULTS.md`, which used the OpenStreetMap grid.

## Using the app

Scenarios are set in the sidebar:

- **Year and length:** start date and days modelled (365 for a full year).
- **Hydrology:** reservoir mode (sustainable or drawdown) and inflow as a share of an average year (0.80 = a 20% drought).
- **Demand:** growth for existing customers, grid-access target, and consumption per new household.
- **Mining:** contract status, and the price at which the optimiser curtails miners.
- **Economics:** discount rate, and whether new capacity may be built.

The tabs show results, hourly dispatch, network flows, inputs, data provenance and diagnostics, and export the solved network.

## Repository layout

| File | Purpose |
|---|---|
| `model.py` | Network builder, solver call and KPIs. No Streamlit import, so it runs headless |
| `app.py` | Streamlit interface |
| `data.py` | Loads data, parses costs, aggregates transmission lines into corridors |
| `hydro.py` | Reservoir rule curves and inflow scaling |
| `timeseries.py` | Averages hourly profiles onto coarser model time steps |
| `setup_data.py` | One-time downloads: costs, shapes, ERA5 cutout, profiles |
| `osm_grid.py` | Finds transmission lines in OpenStreetMap that are missing from the 2007 network data |
| `data/` | Inputs: demand, access, plants, hydro, costs, profiles |
| `RESULTS.md` | Results of the September 2026 analysis |

## Limitations

Six nodes cannot represent transmission constraints within regions, and line capacities are estimated from voltage class. Drought is modelled by scaling one weather year's inflow rather than replaying a historical dry year. Costs are European technology-data values, so local solar and financing costs may differ. See [RESULTS.md](RESULTS.md#caveats) for the full list.

## Author

Kuma Erkisso, M.Sc. Renewable Energy Systems (Hochschule Nordhausen). Feedback and questions are welcome via GitHub issues.
