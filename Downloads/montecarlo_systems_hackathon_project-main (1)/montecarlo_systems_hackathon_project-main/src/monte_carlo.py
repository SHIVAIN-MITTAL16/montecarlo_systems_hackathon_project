"""
src/monte_carlo.py
==================
Monte Carlo scenario engine.
Generates 5,000 stress-tested grid futures using real ERCOT statistical profiles.
"""

import numpy as np
import pandas as pd
import json, os, time


class ScenarioGenerator:

    STRESS_PROB        = 0.15
    STRESS_DEMAND_ADD  = (5_000, 12_000)
    STRESS_SOLAR_MULT  = (0.0, 0.30)
    STRESS_WIND_MULT   = (0.0, 0.40)
    CRISIS_PROB        = 0.03
    CRISIS_DEMAND_MULT = (1.05, 1.12)
    CRISIS_SOLAR_MULT  = 0.05
    CRISIS_WIND_MULT   = 0.10

    # Dispatch parameters
    BASELINE_GAS_FRAC  = 0.90   # baseline: 90% of demand as firm gas
    OPTIMAL_GAS_FRAC   = 0.98   # optimal: 98% of demand
    BATTERY_POWER_MW   = 2_000
    BATTERY_ENERGY_MWH = 8_000
    BATTERY_RTE        = 0.90
    GAS_COST           = 80     # $/MWh
    BATTERY_COST       = 5      # $/MWh
    VOLL               = 5_000  # $/MWh unserved
    GAS_CARBON         = 490    # kg CO2/MWh

    def __init__(self, data_path="data/clean_data.csv"):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Missing: {data_path}\nRun: python setup.py first.")
        df = pd.read_csv(data_path, parse_dates=["datetime"])
        self.real_demand = df["demand_mw"].values
        self.real_solar  = df["solar_mw"].values
        self.real_wind   = df["wind_mw"].values
        self.peak_demand = df["demand_mw"].max()
        self.demand_cap  = self.peak_demand * 1.12
        self.max_solar   = df["solar_mw"].max()
        self.max_wind    = df["wind_mw"].max()

    def generate(self, n=5_000, seed=42):
        np.random.seed(seed)
        t0 = time.time()

        idx    = np.random.randint(0, len(self.real_demand), n)
        demand = self.real_demand[idx].astype(float)
        solar  = self.real_solar[idx].astype(float)
        wind   = self.real_wind[idx].astype(float)

        demand += np.random.normal(0, 2000, n)
        solar  += np.random.normal(0,  400, n)
        wind   += np.random.normal(0,  800, n)

        stress = np.random.rand(n) < self.STRESS_PROB
        crisis = (~stress) & (np.random.rand(n) < self.CRISIS_PROB)

        demand[stress] += np.random.uniform(*self.STRESS_DEMAND_ADD, stress.sum())
        solar[stress]  *= np.random.uniform(*self.STRESS_SOLAR_MULT, stress.sum())
        wind[stress]   *= np.random.uniform(*self.STRESS_WIND_MULT,  stress.sum())

        demand[crisis] *= np.random.uniform(*self.CRISIS_DEMAND_MULT, crisis.sum())
        solar[crisis]  *= self.CRISIS_SOLAR_MULT
        wind[crisis]   *= self.CRISIS_WIND_MULT

        demand = np.clip(demand, 0, self.demand_cap)
        solar  = np.clip(solar,  0, self.max_solar)
        wind   = np.clip(wind,   0, self.max_wind)

        renewable = solar + wind
        gap = np.clip(demand - renewable, 0, None)

        df = pd.DataFrame({
            "demand_mw":    demand.round(1),
            "solar_mw":     solar.round(1),
            "wind_mw":      wind.round(1),
            "renewable_mw": renewable.round(1),
            "gap_mw":       gap.round(1),
            "stress_event": stress,
            "crisis_event": crisis,
        })

        df["risk_label"] = np.where(
            gap > 0.25 * demand, "HIGH",
            np.where(gap > 0.10 * demand, "MEDIUM", "LOW")
        )

        print(f"  Generated {n:,} scenarios in {time.time()-t0:.2f}s")
        return df

    def simulate_baseline(self, scenarios):
        """Baseline: proportional gas cap (90% demand), no battery."""
        results = []
        for _, r in scenarios.iterrows():
            demand   = float(r["demand_mw"])
            gap      = float(r["gap_mw"])
            gas_cap  = self.BASELINE_GAS_FRAC * demand
            gas_used = min(gap, gas_cap)
            unserved = max(0.0, gap - gas_used)
            cost     = gas_used * self.GAS_COST + unserved * self.VOLL
            carbon   = gas_used * self.GAS_CARBON / 1_000
            rel      = max(0.0, 1.0 - unserved / demand) if unserved > 0 else 1.0
            results.append({
                "baseline_cost":           round(cost, 2),
                "baseline_carbon_tonnes":  round(carbon, 2),
                "baseline_reliability":    round(rel, 4),
                "baseline_gas_used":       round(gas_used, 1),
                "baseline_unserved_mw":    round(unserved, 1),
            })
        return pd.DataFrame(results)

    def simulate_optimal(self, scenarios):
        """Optimal: higher gas cap + 2 GW / 8 GWh battery dispatch."""
        results = []
        soc = self.BATTERY_ENERGY_MWH * 0.5

        for _, r in scenarios.iterrows():
            demand    = float(r["demand_mw"])
            renewable = float(r["renewable_mw"])
            gap       = demand - renewable

            bat_used = 0.0
            gas_used = 0.0
            unserved = 0.0

            if gap <= 0:
                surplus  = -gap
                charge   = min(surplus, self.BATTERY_POWER_MW,
                               (self.BATTERY_ENERGY_MWH - soc) / self.BATTERY_RTE)
                soc     += charge * self.BATTERY_RTE
            else:
                discharge = min(gap, self.BATTERY_POWER_MW, soc)
                gap      -= discharge
                soc      -= discharge
                bat_used  = discharge

                gas_cap  = self.OPTIMAL_GAS_FRAC * demand
                gas      = min(gap, gas_cap)
                gap     -= gas
                gas_used = gas

                if gap > 0:
                    unserved = gap

            soc = float(np.clip(soc, 0, self.BATTERY_ENERGY_MWH))

            cost   = gas_used * self.GAS_COST + bat_used * self.BATTERY_COST + unserved * self.VOLL
            carbon = gas_used * self.GAS_CARBON / 1_000
            rel    = max(0.0, 1.0 - unserved / demand) if unserved > 0 else 1.0

            results.append({
                "cost":          round(cost, 2),
                "carbon_tonnes": round(carbon, 2),
                "reliability":   round(rel, 4),
                "gas_used":      round(gas_used, 1),
                "battery_used":  round(bat_used, 1),
                "unserved_mw":   round(unserved, 1),
            })
        return pd.DataFrame(results)


def run_monte_carlo(n=5_000, data_path="data/clean_data.csv",
                    output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Running Monte Carlo engine ({n:,} scenarios)...")

    gen       = ScenarioGenerator(data_path)
    scenarios = gen.generate(n)
    scenarios.to_csv(f"{output_dir}/scenarios.csv", index=False)

    print("Simulating baseline strategy...")
    baseline = gen.simulate_baseline(scenarios)
    baseline.to_csv(f"{output_dir}/baseline_results.csv", index=False)

    print("Simulating optimal strategy...")
    optimal = gen.simulate_optimal(scenarios)
    optimal.to_csv(f"{output_dir}/optimal_results.csv", index=False)

    bl_cost  = baseline["baseline_cost"].mean()
    opt_cost = optimal["cost"].mean()
    bl_carb  = baseline["baseline_carbon_tonnes"].mean()
    opt_carb = optimal["carbon_tonnes"].mean()
    bl_risk  = (baseline["baseline_reliability"] < 1).mean() * 100
    opt_risk = (optimal["reliability"] < 1).mean() * 100

    metrics = {
        "cost_reduction_pct":    round((bl_cost - opt_cost) / bl_cost * 100, 2),
        "carbon_reduction_pct":  round((bl_carb - opt_carb) / bl_carb * 100, 2),
        "baseline_blackout_pct": round(bl_risk,  2),
        "optimal_blackout_pct":  round(opt_risk, 2),
        "risk_reduction_pp":     round(bl_risk - opt_risk, 2),
        "n_scenarios": n,
    }

    import json
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n===== RESULTS =====")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    return {"scenarios": scenarios, "baseline": baseline,
            "optimal": optimal, "metrics": metrics}


if __name__ == "__main__":
    run_monte_carlo()
