# PyPSA-Ethiopia: results

**Can Ethiopia's power system absorb demand growth and drought, and what role does crypto-mining curtailment play?**

September 2026 · Kuma Erkisso · [github.com/kbuni123](https://github.com/kbuni123)

## Summary

- **Mining is the system's shock absorber.** In every scenario, the least-cost response to a supply shortfall is to curtail crypto miners at their tariff (€28.9/MWh) before building anything or cutting other customers.
- **The model brackets EEP's real decision.** In a 2026 drought with 20% less water, the model serves between 4% of miners (reservoirs held level) and 34% (stored water fully used). EEP's actual September 2026 cut to 23% of contracted supply falls inside that range, and only for a water shortfall of roughly 20–25%, which matches the reported drought.
- **Growth or drought alone needs no new capacity by 2030. Both together do.** With 5% annual demand growth and a dry year, the model builds about **1.3 GW of solar** to protect non-mining customers. This investment is insensitive to financing cost.
- **Whether to build solar to keep miners running is a policy question.** Replacing curtailment with solar costs about **€30/MWh** at an 8% discount rate and about **€40/MWh** at 12%. If mining is valued above that, the model builds around **5.5 GW** of solar; below it, none.

## The model

A six-node capacity-expansion and dispatch model of the Ethiopian grid, built with [PyPSA](https://pypsa.org) and solved with HiGHS. The nodes are Addis Ababa, Bahir Dar, Mekelle, Hawassa, Dire Dawa and Jigjiga, connected by corridors aggregated from OpenStreetMap transmission lines.

| Component | Source / treatment |
|---|---|
| Weather | ERA5 reanalysis via atlite: area-weighted solar, wind and runoff profiles per node |
| Costs | PyPSA technology-data, 2030 values |
| Hydropower | Large plants as reservoirs, with inflow scaled to design energy. **Sustainable** mode holds each reservoir to its annual rule curve; **drawdown** mode lets GERD and Beles use stored water (2.77 TWh in total) |
| Demand | Anchored to EEP's FY2025/26 generation of 35,671 GWh. Crypto mining is 27% of domestic sales (EEP), frozen at that level because new permits stopped in August 2025. Kenya exports follow the contract (400/150 MW peak/off-peak from December 2026) |
| New connections | Grid access rising from 29.3% (2025) to 41% by 2030, NEP's target pace. At 365 kWh per household (Tier 3) |
| Mining curtailment | Allowed at the mining tariff, 3.14 US¢/kWh ≈ €28.9/MWh |
| Unserved load | Priced at €3,000/MWh for all other customers |

System cost counts new capacity and operation. Existing plants and lines are treated as sunk.

## Base-year check

Run on FY2025/26 with the existing fleet, the model reproduces EEP's reported generation of 35,671 GWh, with no unserved energy and all miners served.

## Validation against EEP's September 2026 decision

On 15 September 2026, EEP cut crypto miners to 23% of contracted supply, citing a 20% hydro shortfall. The model's 2026 system (no demand growth, Koysha not yet online) was run across drought severities and both reservoir modes:

| Inflow (share of average) | Reservoirs held level | Stored water used | Is EEP's 23% inside? |
|---|---|---|---|
| 0.70 | 0% served, solar built | ~0% | no |
| **0.80** | **4%** | **34%** | **yes** |
| 0.90 | 38% | 69% | no |

At inflow 0.80, the curtailed volume is 8.67 TWh with reservoirs held level and 5.90 TWh with stored water used. Serving 23% of miners requires about 1.7 of the 2.77 TWh of available stored water, roughly 60%. A possible reading is that EEP is drawing reservoirs down while holding back a reserve. The model is consistent with this but cannot confirm it.

**Robustness to mining's share.** Raising mining from 27% to 33% of domestic sales (the upper end of press estimates) leaves the curtailed volume unchanged to the euro, because total demand is fixed by EEP's generation figure. Only the served percentage shifts, to 21–46%. EEP's 23% stays inside the bracket.

## 2030 scenarios

The 2030 system includes Koysha (2,160 MW), commissioned in 2028–29. Demand grows from new connections, the larger Kenya contract, and, where stated, 5% a year for existing customers.

| Scenario | Demand | Miners served | Mining curtailed | New build | System cost |
|---|---|---|---|---|---|
| 0% growth, average water | 38.7 TWh | 100% | none | none | — |
| 5% growth, average water | 43.9 TWh | 55% | 4.1 TWh | none | €21.8 M |
| 0% growth, drought (inflow 0.80) | 38.7 TWh | 29% | 6.4 TWh | none | €18.0 M |
| **5% growth, drought** | **43.9 TWh** | **0%** | **9.0 TWh** | **~1.3 GW solar** | **€96.5 M** |

Koysha is what lifts the drought case from 4% of miners served in 2026 to 29% in 2030 despite higher demand. With both growth and drought, curtailing every miner is not enough: the model builds solar to avoid shedding other customers, and it supplies about 2.8 TWh.

## Sensitivity: what decides the solar build

On the 5% growth, drought scenario:

| Discount rate | Mining curtailment price | New solar | Miners served | System cost |
|---|---|---|---|---|
| 8% | €28.9 (tariff) | ~1.3 GW | 0% | €96.5 M |
| 8% | €40 | ~5.5 GW | 100% | €371.0 M |
| 12%* | €28.9 | unchanged | 0% | higher |
| 12%* | €40 | unchanged | ~1% | higher |

\* The 12% runs used the earlier access assumption (47% by 2030). They show the same pattern: solar capacity identical to the 8% tariff case, only the cost rising.

The solar splits into two parts:

1. **About 1.3 GW of drought insurance.** It protects non-mining customers, whose shortages cost €3,000/MWh, so no plausible financing cost changes the decision. This is the robust investment finding.
2. **About 4 GW that exists only to serve miners.** Its cost per MWh of curtailment avoided is €30.5 at 8% and about €40 at 12%. At €40, it is built at 8% and just fails at 12%. Whether this capacity makes sense depends on how Ethiopia values mining revenue, including its foreign-currency earnings, against the cost of capital.

## Caveats

- **Six nodes.** Transmission within regions is not modelled, and corridor capacities are estimated from voltage class.
- **One weather year.** Drought is represented by scaling annual inflow, not by a historical dry year.
- **Inflow 0.80 is assumed** from the reported 20% shortfall. The validation shows results are sensitive to it.
- **Drawdown mode starts reservoirs full**, which is optimistic after a dry season.
- **Different time periods.** EEP's 23% applies from mid-September; the model reports calendar-year averages.
- **Several inputs have no published source**: regional demand split, hourly load shape, 5% growth, and household consumption. Each is marked as an assumption in `data/*.yaml`.
- **Costs are European technology-data values.** Local costs, particularly for solar and finance, may differ; the discount-rate sensitivity covers part of this.
- **Mining is frozen at the base-year level.** If new permits resumed, the conclusions would change.

## Reproducing these results

Run with `streamlit run app.py`. All runs use 365 days, mining share 0.27 in `data/demand_ethiopia.yaml`, and app defaults unless listed.

| Result | Start | Reservoirs | Growth | Inflow | Other |
|---|---|---|---|---|---|
| 2026 validation | 2026-01-01 | sustainable / drawdown | 0.00 | 0.70 / 0.80 / 0.90 | — |
| 2030, average water | 2030-01-01 | sustainable | 0.00 / 0.05 | 1.00 | — |
| 2030, drought | 2030-01-01 | sustainable | 0.00 / 0.05 | 0.80 | — |
| Sensitivity | 2030-01-01 | sustainable | 0.05 | 0.80 | discount 8% / 12%; mining price €28.9 / €40 |

Leave **Existing system only** unticked and **Grid expansion to new households** ticked. Solar capacities are read from the Network tab.
