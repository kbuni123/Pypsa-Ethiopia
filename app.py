"""
PyPSA-Ethiopia -- Streamlit capacity-expansion app.

Run with:
    streamlit run app.py

All modelling lives in model.py; this file is presentation only.
"""

from __future__ import annotations

import io
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pypsa
import streamlit as st

import data as D
from model import (
    REGIONS,
    TECHS,
    EEP_BASE_YEAR_GWH,
    ScenarioConfig,
    build_network,
    dispatch_frame,
    hydro_profile,
    load_profile,
    make_snapshots,
    solar_profile,
    solve,
    summarise,
    validate_network,
    wind_profile,
)

logging.getLogger("pypsa").setLevel(logging.WARNING)
logging.getLogger("linopy").setLevel(logging.WARNING)

st.set_page_config(page_title="PyPSA-Ethiopia", page_icon="⚡", layout="wide")

# --------------------------------------------------------------------------
# Width-parameter compatibility: Streamlit renamed `use_container_width` to
# `width="stretch"` in 1.49. Detect once so the app runs on old and new alike.
# --------------------------------------------------------------------------
def _stretch(fn) -> dict:
    try:
        params = __import__("inspect").signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


W_DF = _stretch(st.dataframe)
W_PLOT = _stretch(st.plotly_chart)
W_BTN = _stretch(st.button)


CARRIER_COLORS = {
    "solar": "#f5b301",
    "wind": "#4a9fd5",
    "hydro": "#1f6f8b",
    "geothermal": "#b05c3c",
    "diesel": "#6b6b6b",
    "battery": "#7cbf6a",
    "load_shedding": "#d64545",
    "mining_curtailment": "#e8a33d",
}


# --------------------------------------------------------------------------
# Sidebar: scenario definition
# --------------------------------------------------------------------------
def sidebar() -> ScenarioConfig:
    st.sidebar.header("Scenario")

    start = st.sidebar.date_input("Start date", pd.Timestamp("2030-01-01"))
    days = st.sidebar.slider("Days modelled", 7, 365, 365,
                             help="Reservoirs shift water between seasons. Anything short of a "
                                  "full year understates hydro -- a dry-season window can't "
                                  "use wet-season water.")
    res = st.sidebar.select_slider("Time resolution", options=[3, 6, 12, 24], value=12,
                                   format_func=lambda h: f"{h} h",
                                   help="12 h solves a full year with investment in ~40 s.")
    if days < 365:
        st.sidebar.warning("Short horizon: seasonal hydro storage can't work. "
                           "Use 365 days for anything you'd quote.")
    st.sidebar.caption(f"{int(days * 24 / res)} snapshots")

    st.sidebar.divider()
    st.sidebar.subheader("Demand")
    peak = st.sidebar.number_input(
        "National peak demand (MW)", 1000.0, 40000.0, 6500.0, step=250.0,
        help="Ignored when anchored demand is on -- the peak then falls out of "
             "published annual energy and the assumed shape.")

    st.sidebar.divider()
    st.sidebar.subheader("Technologies")
    enabled = {}
    cols = st.sidebar.columns(2)
    for i, key in enumerate(["solar", "wind", "hydro", "geothermal", "diesel", "battery"]):
        enabled[key] = cols[i % 2].checkbox(key.capitalize(), value=True, key=f"en_{key}")

    existing_hydro = st.sidebar.checkbox("Include existing hydro fleet", value=True)
    expand_network = st.sidebar.checkbox("Allow transmission expansion", value=True)

    st.sidebar.divider()
    st.sidebar.subheader("Hydrology & mining")
    hydro_mode = st.sidebar.radio(
        "Reservoirs", ["sustainable", "drawdown"], horizontal=True,
        help="sustainable: each reservoir ends the year where it started (future years). "
             "drawdown: GERD may draw down up to the 2.54 TWh it actually did in FY2025/26.")
    inflow = st.sidebar.slider("Inflow (share of design)", 0.5, 1.2, 1.0, 0.05,
                               help="0.8 is roughly the 2026 El Nino shortfall EEP reported.")
    mining_status = st.sidebar.radio(
        "Mining after EEP's October decision", ["continue", "curtail", "terminate"],
        horizontal=True, help="continue = today's contracted level (frozen: no new permits); "
                              "curtail = 23%, the September 2026 level; terminate = 0.")
    try:
        _tariff = float(D.load_demand_anchors()["mining"]["curtail_price_eur_mwh"])
    except Exception:
        _tariff = 28.9
    mining_price = st.sidebar.slider(
        "Mining curtailment price (EUR/MWh)", 0.0, 150.0, _tariff, 0.5,
        help=f"What the optimiser pays for each MWh of mining it curtails. {_tariff:g} is the "
             "tariff miners pay (3.14 US cents/kWh). Raise it to value the foreign-currency "
             "revenue mining brings. At an 8% discount rate new solar starts replacing "
             "curtailment at roughly 30 EUR/MWh; at 12%, roughly 40.")
    existing_only = st.sidebar.checkbox(
        "Existing system only (no new build)", value=False,
        help="Shows what a shock does to today's system, rather than what you'd build for it.")

    st.sidebar.divider()
    st.sidebar.subheader("Demand growth & access")
    growth = st.sidebar.slider("Underlying growth, existing customers", 0.0, 0.15, 0.05, 0.01,
                               format="%.2f",
                               help="PLACEHOLDER -- no sourced figure. Excludes new connections.")
    include_access = st.sidebar.checkbox("Grid expansion to new households", value=True)
    # Defaults come from data/access_ethiopia.yaml so the app and the data agree.
    try:
        _traj = D.load_access()["trajectory"]
        _acc_target, _acc_year = float(_traj["grid_target"]), int(_traj["grid_target_year"])
    except Exception:
        _acc_target, _acc_year = 0.41, 2030
    access_target = st.sidebar.slider(
        "Grid-connected households, target", 0.30, 1.0, _acc_target, 0.01,
        help=f"29.3% today (Energy Access Survey 2025). Default {_acc_target:.0%} by {_acc_year} "
             "is what NEP's target pace of ~1 million connections a year yields "
             "(access_ethiopia.yaml). At the actual recent pace (~250,000/yr) grid access "
             "stays near 30%. The old default, 65% by 2035, applied NEP 2.0's grid/off-grid "
             "split and gave ~47% in 2030.")
    access_year = st.sidebar.slider("…reached in", 2027, 2045, _acc_year)
    kwh_hh = st.sidebar.select_slider(
        "Use per new household (kWh/yr)", options=[73, 365, 1250, 3000], value=365,
        help="ESMAP Multi-Tier Framework thresholds: Tier 2 = 73, Tier 3 = 365, "
             "Tier 4 = 1,250, Tier 5 = 3,000.")

    st.sidebar.divider()
    st.sidebar.subheader("Economics & policy")
    rate = st.sidebar.slider("Discount rate", 0.02, 0.15, 0.08, 0.005,
                             format="%.3f")
    cap_on = st.sidebar.checkbox("Annual CO₂ cap", value=False)
    co2 = None
    if cap_on:
        co2 = st.sidebar.number_input("Cap (t CO₂ per year)", 0.0, 20_000_000.0,
                                      500_000.0, step=50_000.0)

    st.sidebar.divider()
    st.sidebar.subheader("Data sources")
    use_real_weather = st.sidebar.checkbox("Real weather (atlite/ERA5)", value=True)
    use_real_plants = st.sidebar.checkbox("Real plant fleet (EEP)", value=True)
    use_real_demand = st.sidebar.checkbox("Anchored demand", value=True)
    use_real_costs = st.sidebar.checkbox("technology-data costs", value=True)
    use_real_grid = st.sidebar.checkbox(
        "Real transmission", value=True,
        help="Real lines aggregated into corridors between the six regions.")
    grid_source = st.sidebar.selectbox(
        "Transmission data", ["auto", "osm", "wb2007"],
        format_func={"auto": "Auto (OSM if downloaded)", "osm": "OpenStreetMap (current)",
                     "wb2007": "World Bank 2006-07"}.get,
        help="OSM is surveyed and current. The 2006-07 data predates GERD and the railway; "
             "rows in transmission_additions.csv marked base=wb2007 patch only that one.")
    use_available = st.sidebar.checkbox(
        "Use available, not nameplate, capacity", value=True,
        help="Available = what the plant list reports as operable; nameplate = design rating.")
    st.sidebar.caption("Each falls back to synthetic if the data file is absent. "
                       "The Data tab shows what was actually used.")

    solver = st.sidebar.selectbox("Solver", ["highs", "glpk", "cbc", "gurobi"], index=0)

    return ScenarioConfig(
        start=str(start),
        days=int(days),
        resolution_hours=int(res),
        peak_demand_mw=float(peak),
        discount_rate=float(rate),
        enabled=enabled,
        existing_hydro=bool(existing_hydro),
        co2_limit_t=co2,
        expand_network=bool(expand_network),
        solver=solver,
        use_real_weather=bool(use_real_weather),
        use_real_plants=bool(use_real_plants),
        use_real_demand=bool(use_real_demand),
        use_real_costs=bool(use_real_costs),
        use_available_capacity=bool(use_available),
        use_real_grid=bool(use_real_grid),
        grid_source=grid_source,
        hydro_mode=hydro_mode,
        inflow_factor=float(inflow),
        mining_status=mining_status,
        mining_curtail_price=float(mining_price),
        allow_new_build=not existing_only,
        underlying_growth=float(growth),
        include_access=bool(include_access),
        grid_access_target=float(access_target),
        grid_access_target_year=int(access_year),
        kwh_per_new_household=float(kwh_hh),
    )


# --------------------------------------------------------------------------
def render_validation(checks) -> bool:
    all_ok = all(ok for _, ok, _ in checks)
    rows = [{"Check": name, "": "✅" if ok else "❌", "Detail": detail}
            for name, ok, detail in checks]
    st.dataframe(pd.DataFrame(rows), hide_index=True, **W_DF)
    return all_ok


def carrier_pie(series: pd.Series, title: str, unit: str) -> go.Figure:
    s = series[series > 0]
    fig = px.pie(values=s.values, names=s.index, hole=0.45, title=title,
                 color=s.index, color_discrete_map=CARRIER_COLORS)
    fig.update_traces(texttemplate="%{label}<br>%{percent}",
                      hovertemplate="%{label}: %{value:,.0f} " + unit)
    fig.update_layout(height=380, margin=dict(t=50, b=10, l=10, r=10))
    return fig


# --------------------------------------------------------------------------
def tab_run(cfg: ScenarioConfig) -> None:
    st.subheader("Build and solve")
    st.write(
        "The builder sets snapshots **before** any time series is created, then "
        "validates the network before handing it to the solver."
    )

    if st.button("▶ Run scenario", type="primary", **W_BTN):
        t0 = time.time()
        with st.status("Building network…", expanded=True) as status_box:
            n = build_network(cfg)
            st.write(f"Network built: {len(n.buses)} buses, "
                     f"{len(n.generators)} generators, {len(n.lines)} lines, "
                     f"{len(n.snapshots)} snapshots.")

            checks = validate_network(n)
            ok = all(o for _, o, _ in checks)
            if not ok:
                status_box.update(label="Validation failed", state="error")
                st.session_state["network"] = n
                st.session_state["checks"] = checks
                st.session_state.pop("summary", None)
                return
            st.write("Validation passed.")

            status_box.update(label=f"Solving with {cfg.solver}…")
            try:
                s_status, s_condition = solve(n, cfg.solver)
            except Exception as exc:  # solver missing, licence, etc.
                status_box.update(label="Solver error", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                st.info("HiGHS ships with `highspy` and needs no licence. "
                        "Install it with `pip install highspy`.")
                return

            if s_condition != "optimal":
                status_box.update(label=f"Solver returned '{s_condition}'", state="error")
                st.error(
                    "The model did not solve to optimality. With the load-shedding "
                    "slack generator present this normally means a bad cost or "
                    "capacity bound rather than a genuine shortage."
                )
                return

            status_box.update(label=f"Solved in {time.time() - t0:.1f} s",
                              state="complete")

        st.session_state["network"] = n
        st.session_state["checks"] = checks
        st.session_state["summary"] = summarise(n)
        st.session_state["cfg"] = cfg.to_dict()

    st.divider()
    st.markdown("**Validate against EEP**")
    st.caption("Runs FY2025/26 with today's fleet only, GERD drawdown capped at the "
               "2.54 TWh it actually drew, and no new build. EEP generated 35,734 GWh.")
    if st.button("Run base-year validation", **W_BTN):
        vcfg = ScenarioConfig(start="2026-01-01", days=365, resolution_hours=12,
                              hydro_mode="drawdown", include_access=False,
                              allow_new_build=False, solver=cfg.solver)
        with st.spinner("Solving FY2025/26 with the existing system…"):
            vn = build_network(vcfg)
            solve(vn, vcfg.solver)
            vs = summarise(vn)
        gen = vs["generation_mwh"] / 1000.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelled generation", f"{gen:,.0f} GWh",
                  delta=f"{gen / EEP_BASE_YEAR_GWH * 100 - 100:+.1f}% vs EEP")
        c2.metric("Miners served", f"{vs['mining_served_share']:.0f}%")
        c3.metric("Firm load unserved", f"{vs['unserved_share']:.2f}%")
        prices = vn.buses_t.marginal_price.mean()
        stuck = prices[prices > 1000].index.tolist()
        if stuck:
            st.warning(f"Firm load unserved at {', '.join(stuck)} -- look at the lines into "
                       "those buses before blaming the generation data.")
        st.caption("A gap here points at a specific input: spill at a plant means transmission; "
                   "a national shortfall with no spill means design energies are too low.")

    if "checks" in st.session_state:
        st.divider()
        st.markdown("**Pre-flight checks**")
        render_validation(st.session_state["checks"])

    if "summary" in st.session_state:
        st.success("Results are ready — see the Results and Dispatch tabs.")


def tab_results() -> None:
    if "summary" not in st.session_state:
        st.info("Run a scenario first.")
        return

    s = st.session_state["summary"]

    c1, c2, c3, c4 = st.columns(4)
    # Penalties steer the optimiser but are not money spent -- name each one.
    _pen = []
    if s.get("mining_curtailment_eur", 0) > 1:
        _pen.append(f"€ {s['mining_curtailment_eur']:,.0f} mining curtailment")
    if s.get("voll_penalty_eur", 0) > 1:
        _pen.append(f"€ {s['voll_penalty_eur']:,.0f} unserved-load penalty")
    if not _pen and s.get("penalty_eur", 0) > 1:     # summary from an older model.py
        _pen.append(f"€ {s['penalty_eur']:,.0f} penalties")
    c1.metric("System cost", f"€ {s.get('system_cost_eur', s['total_cost_eur']):,.0f}",
              delta=("+ " + ", + ".join(_pen)) if _pen else None,
              delta_color="off",
              help="Annualised cost of NEW capacity plus operating cost of the whole fleet. "
                   "Existing plants and the existing grid are sunk and excluded "
                   + (f"(grid annuity € {s['existing_grid_annuity_eur']:,.0f} not counted). "
                      if s.get("existing_grid_annuity_eur") else ". ")
                   + "Mining curtailment (at the curtailment price) and unserved load "
                   "(at €3,000/MWh) are penalties that steer the optimiser -- shown "
                   "separately, not spent.")
    c2.metric("New build + operating cost", f"€ {s['lcoe_eur_mwh']:.2f} /MWh",
              help="System cost divided by generation. NOT a full LCOE: existing plants "
                   "and lines carry no capital cost, so with no new build this is close "
                   "to operating cost alone.")
    c3.metric("Renewable share", f"{s['renewable_share']:.1f} %")
    c4.metric("Unserved energy", f"{s['unserved_share']:.2f} %",
              delta=f"{s['unserved_mwh']:,.0f} MWh" if s["unserved_mwh"] >= 0.5 else None,
              delta_color="inverse")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demand", f"{s['demand_mwh'] / 1e6:,.2f} TWh")
    c2.metric("Generation", f"{s['generation_mwh'] / 1e6:,.2f} TWh")
    ms = s.get("mining_served_share")
    c3.metric("Miners served", "—" if ms is None else f"{ms:.0f}%")
    c4.metric("Hydro spilled", f"{s['hydro_spill_mwh'] / 1e6:,.2f} TWh")

    parts = s.get("demand_parts_mwh", {})
    acc = ((s.get("meta") or {}).get("demand") or {}).get("access")
    if parts:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Existing customers", f"{parts.get('non_mining', 0) / 1e6:,.1f} TWh")
        c2.metric("New connections", f"{parts.get('access', 0) / 1e6:,.1f} TWh",
                  help=None if not acc else
                  f"{acc['new_grid_households'] / 1e6:.1f} M households; grid access {acc['grid_access']:.0%}")
        c3.metric("Mining", f"{parts.get('mining', 0) / 1e6:,.1f} TWh")
        c4.metric("Exports", f"{parts.get('exports', 0) / 1e6:,.1f} TWh")
        if acc and acc.get("connection_capex_usd"):
            st.caption(f"Connection capex (not optimised): ${acc['connection_capex_usd'] / 1e9:,.2f} bn")

    n_res = st.session_state.get("network")
    if n_res is not None and s.get("unserved_mwh", 0) > 1:
        w = n_res.snapshot_weightings.objective
        sg = n_res.generators.index[n_res.generators.carrier == "load_shedding"]
        by_bus = (n_res.generators_t.p[sg].mul(w, axis=0).sum()
                  .groupby(n_res.generators.loc[sg, "bus"]).sum() / 1e3)
        by_bus = by_bus[by_bus > 0.5].sort_values(ascending=False)
        if not by_bus.empty:
            st.caption("Unserved energy by bus (GWh) -- where the shortfall actually is:")
            st.dataframe(by_bus.round(0).rename("GWh").to_frame().T, hide_index=True, **W_DF)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(carrier_pie(s["capacity_mw"], "Optimal capacity", "MW"),
                        **W_PLOT)
        st.dataframe(s["capacity_mw"].round(1).rename("MW"), **W_DF)
    with right:
        supply = s["supply_mwh"].drop(labels=["load_shedding"], errors="ignore")
        st.plotly_chart(carrier_pie(supply, "Energy supplied", "MWh"),
                        **W_PLOT)
        st.dataframe(supply.round(0).rename("MWh"), **W_DF)

    st.divider()
    st.markdown("**Cost breakdown (scaled to the modelled horizon)**")
    cost = pd.DataFrame({
        "Annualised CAPEX (€)": s["capex_by_carrier"],
        "OPEX (€)": s["opex_by_carrier"],
    }).fillna(0.0)
    cost["Total (€)"] = cost.sum(axis=1)
    st.dataframe(cost.round(0).sort_values("Total (€)", ascending=False),
                 **W_DF)

    exp = s["line_expansion"]
    exp = exp[exp > 0.1]
    if not exp.empty:
        st.markdown("**Transmission reinforcement (MW added)**")
        st.dataframe(exp.rename("MW added"), **W_DF)


def tab_dispatch() -> None:
    if "network" not in st.session_state or "summary" not in st.session_state:
        st.info("Run a scenario first.")
        return

    n = st.session_state["network"]
    df = dispatch_frame(n)

    gen_cols = [c for c in df.columns if not c.endswith("_net")]
    fig = go.Figure()
    for col in gen_cols:
        if df[col].abs().sum() < 1e-6:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=col, mode="lines",
            stackgroup="gen", line=dict(width=0.5),
            fillcolor=CARRIER_COLORS.get(col),
        ))

    demand = n.loads_t.p_set.sum(axis=1)
    fig.add_trace(go.Scatter(x=demand.index, y=demand.values, name="demand",
                             mode="lines", line=dict(color="black", width=2, dash="dot")))
    fig.update_layout(height=480, hovermode="x unified",
                      yaxis_title="MW", margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig, **W_PLOT)

    st.markdown("**Storage state of charge**")
    if not n.storage_units.empty and not n.storage_units_t.state_of_charge.empty:
        soc = n.storage_units_t.state_of_charge.sum(axis=1)
        if soc.abs().sum() > 1e-6:
            st.area_chart(soc.rename("MWh"), height=220)
        else:
            st.caption("No storage was built in this scenario.")
    else:
        st.caption("Storage is disabled in this scenario.")

    st.divider()
    bus = st.selectbox("Inspect a single node", list(n.buses.index))
    bus_gens = n.generators.index[n.generators.bus == bus]
    node = n.generators_t.p[bus_gens].T.groupby(n.generators.carrier[bus_gens]).sum().T
    hydro_here = n.storage_units.index[(n.storage_units.bus == bus) &
                                        (n.storage_units.carrier == "hydro")]
    if len(hydro_here):
        node["hydro"] = node.get("hydro", 0.0) + n.storage_units_t.p[hydro_here].clip(lower=0).sum(axis=1)
    node["demand"] = n.loads_t.p_set[n.loads.index[n.loads.bus == bus]].sum(axis=1)
    st.line_chart(node, height=300)


def tab_network() -> None:
    if "network" not in st.session_state:
        st.info("Run a scenario first.")
        return
    n = st.session_state["network"]

    fig = go.Figure()
    for _, line in n.lines.iterrows():
        b0, b1 = n.buses.loc[line.bus0], n.buses.loc[line.bus1]
        loading = 0.0
        if not n.lines_t.p0.empty and line.name in n.lines_t.p0:
            s_nom = max(float(line.s_nom_opt or line.s_nom), 1e-6)
            loading = float(n.lines_t.p0[line.name].abs().max()) / s_nom * 100
        fig.add_trace(go.Scattergeo(
            lon=[b0.x, b1.x], lat=[b0.y, b1.y], mode="lines",
            line=dict(width=1 + 4 * min(loading, 100) / 100, color="#4a9fd5"),
            name=line.name, hovertext=f"{line.name}: peak loading {loading:.0f}%",
            showlegend=False,
        ))

    sizes = n.generators.groupby("bus").p_nom_opt.sum().reindex(n.buses.index).fillna(0)
    fig.add_trace(go.Scattergeo(
        lon=n.buses.x, lat=n.buses.y, mode="markers+text",
        text=n.buses.index, textposition="top center",
        marker=dict(size=10 + 25 * sizes / max(sizes.max(), 1e-6), color="#b05c3c"),
        hovertext=[f"{b}: {sizes[b]:,.0f} MW" for b in n.buses.index],
        showlegend=False,
    ))
    fig.update_geos(scope="africa", center=dict(lat=9.5, lon=39.5),
                    projection_scale=6, showcountries=True,
                    landcolor="#f2efe9", countrycolor="#cfc9bd")
    fig.update_layout(height=520, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, **W_PLOT)

    st.markdown("**Lines**")
    cols = ["bus0", "bus1", "length", "s_nom", "s_nom_opt"]
    st.dataframe(n.lines[cols].round(1), **W_DF)


def tab_inputs(cfg: ScenarioConfig) -> None:
    st.subheader("Input profiles")
    st.caption("Deterministic: the same scenario always produces the same series.")

    sn = make_snapshots(cfg)
    region = next(r for r in REGIONS
                  if r.name == st.selectbox("Region", [r.name for r in REGIONS]))

    profiles = pd.DataFrame({
        "solar (p.u.)": solar_profile(sn, region),
        "wind (p.u.)": wind_profile(sn, region),
        "hydro (p.u.)": hydro_profile(sn),
    })
    st.line_chart(profiles, height=280)
    st.line_chart(load_profile(sn, region, cfg.peak_demand_mw).rename("load (MW)"),
                  height=220)

    st.markdown("**Mean capacity factors**")
    st.dataframe(profiles.mean().round(3).rename("p.u."), **W_DF)

    st.divider()
    st.markdown("**Cost assumptions**")
    st.dataframe(pd.DataFrame([
        {"Technology": k, "CAPEX (€/MW)": t.capex, "FOM (%/yr)": t.fom_pct * 100,
         "Lifetime (yr)": t.lifetime, "Marginal (€/MWh)": t.marginal_cost,
         "CO₂ (t/MWh)": t.co2}
        for k, t in TECHS.items()
    ]), hide_index=True, **W_DF)


def tab_diagnostics(cfg: ScenarioConfig) -> None:
    st.subheader("Diagnostics")

    st.markdown("#### Why the earlier version returned zeros")
    st.write(
        "A fresh `pypsa.Network()` has no user snapshots. Calling "
        "`len(n.snapshots)` before `set_snapshots()` therefore produced "
        "length-0 arrays, so every load carried no demand and every generator "
        "had no availability. The solver answered correctly: nothing to serve, "
        "nothing to build, €0."
    )
    if st.button("Reproduce the old bug"):
        broken = pypsa.Network()
        st.code(
            f"n = pypsa.Network()\n"
            f"len(n.snapshots)  # -> {len(broken.snapshots)}\n"
            f"np.random.uniform(100, 500, len(n.snapshots))  "
            f"# -> array of length {len(np.random.uniform(100, 500, 0))}",
            language="python",
        )
        st.error("Empty p_set and p_max_pu → zero demand, zero generation, €0 objective.")
        st.success("Fix: call `n.set_snapshots(...)` first, then build every profile "
                   "against that index.")

    st.divider()
    st.markdown("#### Solver smoke test")
    st.caption("Two buses, one day, one gas unit — confirms PyPSA and the solver work.")
    if st.button("Run smoke test"):
        t = pypsa.Network()
        idx = pd.date_range("2030-01-01", periods=24, freq="h")
        t.set_snapshots(idx)
        t.add("Bus", "b")
        t.add("Carrier", "gas")
        t.add("Load", "l", bus="b", p_set=pd.Series(np.linspace(100, 300, 24), index=idx))
        t.add("Generator", "g", bus="b", carrier="gas", p_nom_extendable=True,
              capital_cost=800.0, marginal_cost=50.0)
        try:
            t.optimize(solver_name=cfg.solver)
            st.write({
                "objective (€)": round(float(t.objective), 2),
                "capacity (MW)": round(float(t.generators.p_nom_opt.iloc[0]), 2),
                "generation (MWh)": round(float(t.generators_t.p.sum().sum()), 2),
            })
            st.success("PyPSA and the solver are working.")
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

    if "network" in st.session_state:
        st.divider()
        st.markdown("#### Component tables")
        n = st.session_state["network"]
        which = st.selectbox("Component", ["generators", "loads", "lines", "buses",
                                           "storage_units", "global_constraints"])
        st.dataframe(getattr(n, which), **W_DF)

        st.markdown("#### Time-series sums (should all be non-zero)")
        sums = {
            "loads_t.p_set": float(n.loads_t.p_set.sum().sum()),
            "generators_t.p_max_pu": float(n.generators_t.p_max_pu.sum().sum())
            if "p_max_pu" in n.generators_t else 0.0,
            "generators_t.p": float(n.generators_t.p.sum().sum())
            if not n.generators_t.p.empty else 0.0,
        }
        st.dataframe(pd.Series(sums).round(1).rename("sum"), **W_DF)


def tab_data() -> None:
    st.subheader("Where the numbers come from")
    st.write(
        "Every input is either sourced or assumed. This table is the honest "
        "answer to 'is this real?' for the run you are about to do."
    )

    prov = D.provenance()
    st.dataframe(prov, hide_index=True, **W_DF)

    synthetic = prov[prov[""] == "synthetic"]
    if not synthetic.empty:
        weather_missing = synthetic["Input"].isin(
            ["Solar profiles", "Wind profiles", "Hydro inflow"]).any()
        hint = (" Run `python setup_data.py --cutout` then `--profiles` for the "
                "weather rows." if weather_missing else
                " The rest are assumptions where no public data exists -- they "
                "are listed so you can state them, not fix them.")
        st.info(f"{len(synthetic)} of {len(prov)} inputs are assumed rather than "
                f"sourced.{hint}")

    st.divider()
    anchors = D.load_demand_anchors()
    if anchors is not None:
        st.markdown("**Demand anchoring**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Base-year generation", f"{anchors['base_year']['total_generation_gwh']:,} GWh",
                  help=f"EEP, FY{anchors['base_year']['fiscal_year']}. Mining "
                       f"{anchors['mining']['share_of_domestic']:.0%} of domestic.")
        c2.metric("Implied peak", f"{D.implied_peak_mw(anchors):,.0f} MW",
                  help="Falls out of the shape -- not a published figure.")
        c3.metric("Implied load factor", f"{D.implied_load_factor(anchors):.3f}")
        st.caption(
            "Annual energy is anchored to EEP's published total. Mining and exports are "
            "flat; only the rest follows the assumed evening-peaking shape -- Ethiopia "
            "does not publish hourly system load."
        )

    plants = D.load_plants()
    if plants is not None:
        st.divider()
        st.markdown("**Generation fleet** (EEP published list)")
        summary = plants.groupby("carrier")[["capacity_mw", "available_mw"]].sum()
        summary.columns = ["Nameplate (MW)", "Available (MW)"]
        st.dataframe(summary.round(1), **W_DF)
        st.caption(
            "Nameplate hydro is about 9,220 MW; only around 5,270 MW was "
            "actually available in 2025, mostly because GERD was running at "
            "roughly 2,350 of its 5,150 MW. Neither column reconciles exactly "
            "with published totals -- see the notes in plants_ethiopia.csv."
        )
        with st.expander("Full plant list"):
            st.dataframe(plants, hide_index=True, **W_DF)


    if D.TRANSMISSION_PATH.exists() and D.GADM_PATH.exists():
        st.divider()
        st.markdown("**Transmission corridors** (real lines aggregated between regions)")
        try:
            buses = pd.DataFrame({"x": [r.lon for r in REGIONS], "y": [r.lat for r in REGIONS]},
                                 index=[r.name for r in REGIONS])
            corridors, rep = D.aggregate_corridors(buses)
            st.caption(f"Source: {rep.get('source', '?')}")
        except Exception as exc:
            st.error(f"Could not aggregate: {type(exc).__name__}: {exc}")
            corridors, rep = None, {}
        if corridors is not None:
            show = corridors[["bus0", "bus1", "s_nom", "n_lines", "voltages", "lines"]].copy()
            show.insert(3, "usable_mw", show.s_nom * D.S_MAX_PU)
            st.dataframe(show.round(0), hide_index=True, **W_DF)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lines in file", rep.get("lines_total", 0))
            c2.metric("Cross regions", rep.get("crossing", 0))
            c3.metric("Internal", len(rep.get("internal", [])))
            c4.metric("Excluded / external",
                      sum(rep.get("excluded_by_reason", {}).values()) + len(rep.get("external", [])))
            if rep.get("isolated_buses"):
                st.warning(
                    f"Not connected in the 2006-07 data: {', '.join(rep['isolated_buses'])}. "
                    "The model patches these with the synthetic corridor so it can run. "
                    "Add the real line to data/transmission_additions.csv if you know it."
                )
            st.caption(
                "Topology is real. Capacity is estimated from voltage class ("
                + ", ".join(f"{k} kV = {v:.0f} MW" for k, v in D.CIRCUIT_MW.items())
                + f"), single circuit assumed, derated to {D.S_MAX_PU:.0%}. "
                "The data predates GERD, so the GERD-to-Addis corridor is missing "
                "until you add it."
            )


    hydro = D.load_hydro_plants()
    if hydro is not None:
        st.divider()
        st.markdown("**Hydro fleet** — reservoirs fed by ERA5 runoff, scaled to design energy")
        show = hydro[["name", "bus", "capacity_mw", "design_gwh", "energy_basis",
                      "storage_months", "online_year", "full_year", "source"]]
        st.dataframe(show, hide_index=True, **W_DF)
        st.caption("Rows marked observed_2016_17 use EEP's 2016/17 capacity factor as a FLOOR "
                   "(a demand-depressed year). Storage months are tiered assumptions.")

    acc = D.load_access()
    if acc is not None:
        st.divider()
        st.markdown("**Electricity access** — the rate depends on the definition")
        sv = acc["survey_2025"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Any source (households)", f"{sv['any_source_households']:.1%}")
        c2.metric(f"World Bank {acc.get('world_bank_year', 2023)} (population)", f"{acc['world_bank_2023_population']:.1%}")
        c3.metric("Tier 1+ (population)", f"{sv['tier1_plus_population']:.0%}")
        c4.metric("On the grid (households)", f"{sv['grid_households']:.1%}")
        st.caption("Only grid connections load the transmission system, so 29.3% is the "
                   "number the model builds from. Off-grid solar (35.7%) is served outside it.")
        try:
            buses = pd.DataFrame({"x": [r.lon for r in REGIONS], "y": [r.lat for r in REGIONS]},
                                 index=[r.name for r in REGIONS])
            weights, wrep = D.access_bus_weights(buses, acc)
            st.markdown("Where new connections land (share of unconnected population):")
            st.dataframe(weights.round(3).rename("share"), **W_DF)
            st.caption(wrep.get("method", ""))
        except Exception as exc:
            st.caption(f"Allocation unavailable: {exc}")

    costs = D.real_tech_costs()
    if costs:
        st.divider()
        st.markdown("**Technology costs** (technology-data)")
        st.dataframe(pd.DataFrame(costs).T, **W_DF)


def tab_export() -> None:
    if "network" not in st.session_state or "summary" not in st.session_state:
        st.info("Run a scenario first.")
        return
    n = st.session_state["network"]
    s = st.session_state["summary"]

    kpis = pd.Series({k: v for k, v in s.items()
                      if isinstance(v, (int, float))}).rename("value")
    st.download_button("KPIs (CSV)", kpis.to_csv().encode(),
                       "pypsa_ethiopia_kpis.csv", "text/csv")
    st.download_button("Optimal capacity (CSV)",
                       s["capacity_mw"].to_csv().encode(),
                       "pypsa_ethiopia_capacity.csv", "text/csv")
    st.download_button("Dispatch (CSV)",
                       dispatch_frame(n).to_csv().encode(),
                       "pypsa_ethiopia_dispatch.csv", "text/csv")

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        n.export_to_netcdf(tmp.name)
        data = Path(tmp.name).read_bytes()
    st.download_button("Solved network (NetCDF)", data,
                       "pypsa_ethiopia.nc", "application/x-netcdf")
    st.caption("Reload with `pypsa.Network('pypsa_ethiopia.nc')`.")


# --------------------------------------------------------------------------
def main() -> None:
    st.title("⚡ PyPSA-Ethiopia")
    st.caption("Capacity expansion and dispatch for the Ethiopian power system — "
               "six nodes, no Snakemake, no external data bundle.")

    cfg = sidebar()

    tabs = st.tabs(["Run", "Results", "Dispatch", "Network", "Inputs",
                    "Data", "Diagnostics", "Export"])
    with tabs[0]:
        tab_run(cfg)
    with tabs[1]:
        tab_results()
    with tabs[2]:
        tab_dispatch()
    with tabs[3]:
        tab_network()
    with tabs[4]:
        tab_inputs(cfg)
    with tabs[5]:
        tab_data()
    with tabs[6]:
        tab_diagnostics(cfg)
    with tabs[7]:
        tab_export()


if __name__ == "__main__":
    main()
