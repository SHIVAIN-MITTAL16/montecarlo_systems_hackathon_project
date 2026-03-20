"""
src/optimizer.py
================
Linear Programming grid capacity optimizer using PuLP.
Finds minimum-cost energy mix with:
  - Capital cost (annualized $/MW-year)
  - Operating cost ($/MWh dispatched)
  - Carbon pricing
  - Battery physics (MW + MWh + SOC + round-trip losses)
  - Firm capacity / reliability constraint
  - Per-scenario capacity factor dispatch
"""

from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum,
    value, LpStatus, PULP_CBC_CMD
)
import pandas as pd
import numpy as np
import os


# ── Technology parameters (industry-calibrated) ──────────────────────────────
PARAMS = dict(
    # Annualized CAPEX ($/MW-year)
    solar_capex      = 60_000,
    wind_capex       = 80_000,
    gas_capex        = 50_000,
    battery_p_capex  = 30_000,   # power (MW)
    battery_e_capex  = 10_000,   # energy (MWh)

    # Variable costs ($/MWh)
    gas_fuel         = 40,
    gas_vom          = 5,
    battery_vom      = 2,

    # Emissions
    carbon_price     = 40,       # $/tonne CO₂
    gas_emission     = 0.49,     # tonne/MWh

    # Reliability
    VOLL             = 5_000,    # $/MWh unserved
    curtail_penalty  = 0.5,

    # Battery physics
    battery_rte      = 0.90,
    battery_duration = 4,
    battery_max_soc  = 0.95,
    battery_min_soc  = 0.05,

    # Capacity credits
    solar_cc         = 0.10,
    wind_cc          = 0.15,
    battery_cc       = 1.00,
    gas_cc           = 0.95,
    reserve_margin   = 0.15,
)


def gas_cost_per_mwh(p: dict) -> float:
    return p["gas_fuel"] + p["gas_vom"] + p["gas_emission"] * p["carbon_price"]


def run_optimizer(csv_path: str = "data/optimizer_input.csv",
                  params: dict = None,
                  msg: int = 0) -> dict:
    """
    Run LP optimizer on scenario CSV.
    CSV must have: demand_mw, solar_cf, wind_cf columns.
    Returns dict with capacity decisions + LCOE.
    """
    if params is None:
        params = PARAMS.copy()
    p = params

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run prepare_optimizer_input() first.")

    df = pd.read_csv(csv_path)
    N  = len(df)

    # Validate
    for col in ["demand_mw", "solar_cf", "wind_cf"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    peak_demand       = df["demand_mw"].max()
    total_demand_mwh  = df["demand_mw"].sum()
    hours_per_scenario= 8760 / N

    model = LpProblem("GridCapacityOptimizer", LpMinimize)

    # ── Capacity variables ────────────────────────────────────────────────────
    solar_cap  = LpVariable("solar_cap",  lowBound=0)
    wind_cap   = LpVariable("wind_cap",   lowBound=0)
    gas_cap    = LpVariable("gas_cap",    lowBound=0)
    bat_power  = LpVariable("bat_power",  lowBound=0)
    bat_energy = LpVariable("bat_energy", lowBound=0)

    # Battery duration constraint
    model += bat_energy == p["battery_duration"] * bat_power

    # Reliability / planning reserve constraint
    model += (
        p["solar_cc"]   * solar_cap +
        p["wind_cc"]    * wind_cap  +
        p["gas_cc"]     * gas_cap   +
        p["battery_cc"] * bat_power
        >= peak_demand * (1 + p["reserve_margin"])
    )

    # ── CAPEX ─────────────────────────────────────────────────────────────────
    capex = (
        p["solar_capex"]     * solar_cap  +
        p["wind_capex"]      * wind_cap   +
        p["gas_capex"]       * gas_cap    +
        p["battery_p_capex"] * bat_power  +
        p["battery_e_capex"] * bat_energy
    )

    # ── Per-scenario dispatch ─────────────────────────────────────────────────
    total_opex = 0
    soc_prev   = None

    for i, row in df.iterrows():
        demand   = float(row["demand_mw"])
        solar_cf = float(row["solar_cf"])
        wind_cf  = float(row["wind_cf"])

        solar_avail = solar_cf * solar_cap
        wind_avail  = wind_cf  * wind_cap

        solar_gen = LpVariable(f"sg_{i}", lowBound=0)
        wind_gen  = LpVariable(f"wg_{i}", lowBound=0)
        gas_gen   = LpVariable(f"gg_{i}", lowBound=0)
        bat_dis   = LpVariable(f"bd_{i}", lowBound=0)
        bat_chg   = LpVariable(f"bc_{i}", lowBound=0)
        unserved  = LpVariable(f"un_{i}", lowBound=0)
        curtail   = LpVariable(f"cu_{i}", lowBound=0)
        soc       = LpVariable(f"soc_{i}", lowBound=0)

        model += solar_gen <= solar_avail
        model += wind_gen  <= wind_avail
        model += gas_gen   <= gas_cap
        model += bat_dis   <= bat_power
        model += bat_chg   <= bat_power
        model += soc       <= p["battery_max_soc"] * bat_energy
        model += soc       >= p["battery_min_soc"] * bat_energy

        if soc_prev is None:
            model += soc == 0.5 * bat_energy + bat_chg * p["battery_rte"] - bat_dis
        else:
            model += soc == soc_prev + bat_chg * p["battery_rte"] - bat_dis

        soc_prev = soc

        model += curtail >= solar_gen + wind_gen + gas_gen + bat_dis - demand
        model += (solar_gen + wind_gen + gas_gen + bat_dis + unserved
                  == demand + bat_chg + curtail)

        opex_i = hours_per_scenario * (
            gas_gen   * gas_cost_per_mwh(p) +
            bat_dis   * p["battery_vom"]    +
            bat_chg   * p["battery_vom"]    +
            unserved  * p["VOLL"]           +
            curtail   * p["curtail_penalty"]
        )
        total_opex += opex_i

    model += capex + total_opex

    # ── Solve ─────────────────────────────────────────────────────────────────
    model.solve(PULP_CBC_CMD(msg=msg))

    annual_demand = total_demand_mwh * (8760 / N)
    obj           = value(model.objective)

    result = {
        "status":                LpStatus[model.status],
        "solar_capacity_mw":     round(value(solar_cap)  or 0, 1),
        "wind_capacity_mw":      round(value(wind_cap)   or 0, 1),
        "gas_capacity_mw":       round(value(gas_cap)    or 0, 1),
        "battery_power_mw":      round(value(bat_power)  or 0, 1),
        "battery_energy_mwh":    round(value(bat_energy) or 0, 1),
        "total_system_cost":     round(obj               or 0, 0),
        "approx_lcoe_per_mwh":   round((obj / annual_demand) if annual_demand else 0, 2),
    }

    return result


def prepare_optimizer_input(scenarios_path: str = "data/scenarios.csv",
                             output_path:   str = "data/optimizer_input.csv") -> str:
    """Convert scenarios.csv → optimizer_input.csv with capacity factors."""
    df = pd.read_csv(scenarios_path)

    solar_cap = max(df["solar_mw"].max(), 1)
    wind_cap  = max(df["wind_mw"].max(),  1)

    df["solar_cf"] = df["solar_mw"] / solar_cap
    df["wind_cf"]  = df["wind_mw"]  / wind_cap

    out = df[["demand_mw", "solar_cf", "wind_cf"]].copy()
    out.to_csv(output_path, index=False)
    print(f"optimizer_input.csv saved ({len(out)} rows)")
    return output_path


if __name__ == "__main__":
    prepare_optimizer_input()
    result = run_optimizer()
    print("\nOptimal Capacity Plan:")
    for k, v in result.items():
        print(f"  {k:<30} {v}")
