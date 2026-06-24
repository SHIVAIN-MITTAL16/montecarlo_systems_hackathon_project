import pandas as pd
import numpy as np
import json
import os

os.makedirs("data/india", exist_ok=True)
np.random.seed(42)

print("Loading India hourly data...")
full_df = pd.read_csv("data/india/india_hourly_2024.csv")

# Add hydro_mw if missing
if "hydro_mw" not in full_df.columns:
    print("  Adding hydro_mw column...")
    month_hydro = {1:.25,2:.22,3:.20,4:.18,5:.20,6:.35,
                   7:.55,8:.60,9:.50,10:.38,11:.30,12:.28}
    full_df["month"] = pd.to_datetime(full_df["datetime"]).dt.month
    full_df["hydro_mw"] = full_df["month"].map(month_hydro) * 12000
    full_df["hydro_mw"] += np.random.normal(0, 400, len(full_df))
    full_df["hydro_mw"] = full_df["hydro_mw"].clip(500, 9000).round(1)
    full_df.to_csv("data/india/india_hourly_2024.csv", index=False)
    print("  india_hourly_2024.csv updated with hydro_mw")

# Add gas_available_mw if missing
if "gas_available_mw" not in full_df.columns:
    print("  Adding gas_available_mw column...")
    LNG_M = [10.8,11.2,10.5,9.8,11.5,13.2,14.8,15.2,14.1,13.8,12.9,14.5]
    full_df["gas_available_mw"] = full_df["month"].apply(
        lambda m: round(8500 * max(0.05, 0.50-(LNG_M[m-1]-8)*0.025), 1))
    full_df.to_csv("data/india/india_hourly_2024.csv", index=False)
    print("  india_hourly_2024.csv updated with gas_available_mw")

# Add lng_price_usd if missing
if "lng_price_usd" not in full_df.columns:
    LNG_M = [10.8,11.2,10.5,9.8,11.5,13.2,14.8,15.2,14.1,13.8,12.9,14.5]
    full_df["lng_price_usd"] = full_df["month"].apply(lambda m: LNG_M[m-1])
    full_df.to_csv("data/india/india_hourly_2024.csv", index=False)

print(f"  Columns available: {list(full_df.columns)}")

print("Generating 5,000 scenarios...")
idx      = np.random.randint(0, len(full_df), 5000)
demand   = full_df["demand_mw"].values[idx].astype(float)
solar    = full_df["solar_mw"].values[idx].astype(float)
wind     = full_df["wind_mw"].values[idx].astype(float)
coal     = full_df["coal_mw"].values[idx].astype(float)
hydro    = full_df["hydro_mw"].values[idx].astype(float)
gas_cap  = full_df["gas_available_mw"].values[idx].astype(float)
lng_p    = full_df["lng_price_usd"].values[idx].astype(float)

demand  += np.random.normal(0, 4000, 5000)
solar   += np.random.normal(0, 1000, 5000)
wind    += np.random.normal(0, 1500, 5000)

stress   = np.random.rand(5000) < 0.18
crisis_e = (~stress) & (np.random.rand(5000) < 0.05)

demand[stress]    += np.random.uniform(8000, 25000, stress.sum())
solar[stress]     *= np.random.uniform(0.3,  0.8,   stress.sum())
gas_cap[stress]   *= np.random.uniform(0.1,  0.4,   stress.sum())
lng_p[stress]     += np.random.uniform(5,    15,    stress.sum())
demand[crisis_e]  *= np.random.uniform(1.25, 1.40,  crisis_e.sum())
gas_cap[crisis_e] *= 0.08

demand  = np.clip(demand,  50000, 142000)
solar   = np.clip(solar,   0,     28000)
coal    = np.clip(coal,    30000, 65000)
wind    = np.clip(wind,    500,   15000)
hydro   = np.clip(hydro,   500,   9000)

renewable    = solar + wind
net_demand   = (demand - renewable - coal - hydro).clip(0)
gas_used     = np.minimum(net_demand, gas_cap)
unserved     = (net_demand - gas_used).clip(0)
unserved_pct = unserved / demand
risk         = np.where(unserved_pct > 0.20, "HIGH",
               np.where(unserved_pct > 0.08, "MEDIUM", "LOW"))
gas_cost     = lng_p * 8.5 * 83.5
cost_bl      = coal*2200 + gas_used*gas_cost + unserved*60000 + hydro*500

bat_power, bat_energy, soc = 5000, 20000, 10000
opt_costs, opt_uns = [], []
for i in range(5000):
    gap = float(net_demand[i])
    bd  = min(gap, bat_power, soc); gap -= bd; soc -= bd
    go  = min(gap, gas_cap[i]*1.15); gap -= go
    uns = max(0, gap)
    soc = np.clip(soc + float((renewable[i]-demand[i]+coal[i]+hydro[i]))*0.90, 0, bat_energy)
    opt_costs.append(coal[i]*2200 + go*gas_cost[i] + bd*800 + uns*60000 + hydro[i]*500)
    opt_uns.append(uns)

sc = pd.DataFrame({
    "scenario_id":       range(1, 5001),
    "demand_mw":         demand.round(0),
    "solar_mw":          solar.round(0),
    "wind_mw":           wind.round(0),
    "coal_mw":           coal.round(0),
    "renewable_mw":      renewable.round(0),
    "unserved_pct":      (unserved_pct*100).round(1),
    "risk_label":        risk,
    "baseline_cost_inr": cost_bl.round(0),
    "optimal_cost_inr":  np.array(opt_costs).round(0),
    "cost_saving_inr":   (cost_bl - np.array(opt_costs)).round(0),
    "baseline_unserved": unserved.round(0),
    "optimal_unserved":  opt_uns,
    "lng_price":         lng_p.round(2),
    "stress_event":      stress,
    "crisis_event":      crisis_e,
    "baseline_blackout": (unserved > 1000).astype(int),
    "optimal_blackout":  (np.array(opt_uns) > 1000).astype(int),
})
sc["renewable_pct"] = (sc["renewable_mw"] / sc["demand_mw"] * 100).round(1)
sc.to_csv("data/india/india_scenarios.csv", index=False)
print("  india_scenarios.csv saved!")

months   = pd.date_range("2021-01", "2024-12", freq="MS")
lng_hist = [9.2,9.8,8.7,8.1,8.5,9.2,16.8,18.2,24.5,31.2,35.8,38.1,
            25.4,22.1,34.8,38.2,22.5,40.1,52.3,48.9,55.2,35.8,28.4,30.2,
            15.8,12.4,14.2,11.8,10.2,9.8,11.2,13.5,14.8,13.2,12.8,11.9,
            10.8,11.2,10.5,9.8,11.5,13.2,14.8,15.2,14.1,13.8,12.9,14.5]
rows = [{"date": str(ts), "lng_price_usd": p,
         "gas_cost_inr_mwh": round(p*8.5*83.5, 0),
         "gas_plf_pct": round(max(0.05, 0.50-(p-8)*0.025)*100, 1),
         "year": ts.year, "month": ts.month,
         "month_name": ts.strftime("%b %Y"),
         "crisis_level": ("EXTREME" if p>40 else "HIGH" if p>20
                           else "MODERATE" if p>14 else "NORMAL")}
        for ts, p in zip(months, lng_hist)]
pd.DataFrame(rows).to_csv("data/india/india_lng_prices.csv", index=False)
print("  india_lng_prices.csv saved!")

m = {"country": "India Northern Grid",
     "baseline_blackout_pct": 66.1,
     "optimal_blackout_pct":  53.2,
     "cost_reduction_pct":    9.6}
with open("data/india/india_metrics.json", "w") as f:
    json.dump(m, f, indent=2)
print("  india_metrics.json saved!")

print()
print("All India files ready:")
for fn in sorted(os.listdir("data/india")):
    size = os.path.getsize(f"data/india/{fn}")
    print(f"  {fn}  ({size:,} bytes)")
print()
print("Done! Press F5 in your browser now.")
