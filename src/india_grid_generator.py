"""
india_grid_generator.py
========================
Generates realistic India Northern Grid (NRGP) dataset
Calibrated to POSOCO 2023, CEA LGBR 2023, MoP data
Covers full year 2024 with crisis window (May-June 2024 heat wave)
"""

import numpy as np
import pandas as pd
import json
import os

np.random.seed(42)

# ── Grid parameters (POSOCO / CEA calibrated) ────────────────────────────────
DEMAND_MEAN    = 87_500   # MW Northern Region average
DEMAND_STD     = 14_200
DEMAND_PEAK    = 140_000  # MW absolute peak
DEMAND_MIN     = 55_000
SOLAR_MEAN     = 12_000   # MW average generation (55 GW installed, ~22% CF)
SOLAR_MAX      = 28_000
WIND_MEAN      = 6_500    # MW average (Rajasthan/Gujarat wind)
WIND_STD       = 3_200
WIND_MIN       = 800
WIND_MAX       = 15_000
GAS_INSTALLED  = 8_500    # MW gas plants (low PLF due to price)
COAL_CAPACITY  = 52_000   # MW coal (backbone)

# LNG price (JKM spot, $/MMBtu) monthly 2024
LNG_MONTHLY_2024 = [10.8,11.2,10.5,9.8,11.5,13.2,14.8,15.2,14.1,13.8,12.9,14.5]
# Gas cost per MWh = LNG price * heat rate / 1000 (heat rate ~8.5 MMBtu/MWh)
GAS_HEAT_RATE   = 8.5
COAL_COST_MWH   = 2200    # ₹/MWh (domestic coal, cheaper)
VOLL_INDIA      = 60_000  # ₹/MWh (value of lost load - lower than US)
CO2_GAS         = 490     # kg/MWh
CO2_COAL        = 820     # kg/MWh


def hour_demand_factor(hour: int, month: int) -> float:
    """India-specific demand curve: evening peak, morning secondary peak."""
    # India peak is 20:00-22:00 (cooking + lighting + AC cooling down)
    # Morning peak 9:00-11:00 (industrial startup)
    base = 0.72
    if 0 <= hour < 5:
        base = 0.68 + 0.02 * hour / 5
    elif 5 <= hour < 9:
        base = 0.70 + 0.10 * (hour - 5) / 4
    elif 9 <= hour < 12:
        base = 0.88 + 0.06 * (hour - 9) / 3    # morning industrial peak
    elif 12 <= hour < 17:
        base = 0.82 + 0.03 * (hour - 12) / 5   # afternoon lull (hot, less activity)
    elif 17 <= hour < 21:
        base = 0.85 + 0.18 * (hour - 17) / 4   # evening ramp - biggest peak
    else:
        base = 0.95 - 0.27 * (hour - 21) / 3   # evening fall-off

    # Seasonal adjustments
    if month in [4, 5, 6]:    # Summer: massive AC load
        base *= 1.22
    elif month in [12, 1]:    # Winter: heating + festival
        base *= 1.08
    elif month in [7, 8, 9]:  # Monsoon: slightly cooler, less AC
        base *= 0.94

    return base


def hour_solar_factor(hour: int, month: int) -> float:
    """India solar: strong in summer, weaker in monsoon (clouds)."""
    if hour < 6 or hour > 19:
        return 0.0
    peak = np.sin((hour - 6) * np.pi / 13)
    # Monthly capacity factor adjustment
    monthly_cf = {1:0.85, 2:0.90, 3:0.95, 4:1.00, 5:1.02,
                  6:0.70, 7:0.60, 8:0.65, 9:0.75, 10:0.88, 11:0.90, 12:0.82}
    return peak * monthly_cf.get(month, 0.85)


def hour_wind_factor(hour: int, month: int) -> float:
    """India wind: monsoon peak Jun-Sep, night stronger in Rajasthan."""
    base = 0.75 + 0.25 * np.cos((hour - 2) * np.pi / 12)
    monthly_wf = {1:0.65, 2:0.60, 3:0.58, 4:0.55, 5:0.60,
                  6:1.05, 7:1.20, 8:1.15, 9:1.00, 10:0.75, 11:0.68, 12:0.65}
    return base * monthly_wf.get(month, 0.75)


def gas_cost_per_mwh(month: int) -> float:
    """Gas cost in ₹/MWh based on LNG spot price."""
    lng = LNG_MONTHLY_2024[month - 1]  # $/MMBtu
    cost_usd = lng * GAS_HEAT_RATE     # $/MWh
    cost_inr = cost_usd * 83.5         # USD→INR at 2024 rate
    return cost_inr


def generate_full_year() -> pd.DataFrame:
    """Generate 8,760 hours of India Northern Grid 2024 data."""
    timestamps = pd.date_range("2024-01-01 00:00", periods=8760, freq="h")
    rows = []

    for ts in timestamps:
        h = ts.hour
        m = ts.month
        doy = ts.day_of_year

        # Demand
        seasonal_demand = DEMAND_MEAN
        if m in [4, 5, 6]:   seasonal_demand += 22_000
        elif m in [12, 1]:    seasonal_demand += 7_000
        elif m in [7, 8, 9]:  seasonal_demand -= 5_000

        demand = (seasonal_demand * hour_demand_factor(h, m)
                  + np.random.normal(0, 3500))
        demand = float(np.clip(demand, DEMAND_MIN, DEMAND_PEAK))

        # Solar
        solar_raw = (SOLAR_MEAN * hour_solar_factor(h, m) * 2.2
                     + np.random.normal(0, 800))
        solar = float(np.clip(solar_raw, 0, SOLAR_MAX))

        # Wind (monsoon-boosted)
        wind_seasonal = WIND_MEAN * hour_wind_factor(h, m)
        wind = float(np.clip(wind_seasonal + np.random.normal(0, 1200),
                              WIND_MIN, WIND_MAX))

        # Coal dispatch (always runs as baseload)
        coal_gen = float(np.clip(COAL_CAPACITY * 0.72
                                  + np.random.normal(0, 2000),
                                  30_000, COAL_CAPACITY))

        # Gas available (PLF depends on price)
        lng_price = LNG_MONTHLY_2024[m - 1]
        gas_plf   = max(0.05, 0.50 - (lng_price - 8) * 0.025)  # PLF drops as LNG rises
        gas_avail = float(GAS_INSTALLED * gas_plf)

        renewable  = solar + wind
        net_demand = max(0, demand - renewable - coal_gen)
        gas_needed = float(np.clip(net_demand, 0, gas_avail))
        unserved   = float(max(0, net_demand - gas_needed))

        gas_cost = gas_cost_per_mwh(m)
        cost_hr  = round(coal_gen * COAL_COST_MWH + gas_needed * gas_cost
                          + unserved * VOLL_INDIA, 0)
        carbon   = round((coal_gen * CO2_COAL + gas_needed * CO2_GAS) / 1000, 2)
        renew_pct= round(renewable / demand * 100, 2)

        rows.append({
            "datetime":        ts,
            "demand_mw":       round(demand, 1),
            "solar_mw":        round(solar, 1),
            "wind_mw":         round(wind, 1),
            "renewable_mw":    round(renewable, 1),
            "coal_mw":         round(coal_gen, 1),
            "gas_available_mw":round(gas_avail, 1),
            "gas_used_mw":     round(gas_needed, 1),
            "unserved_mw":     round(unserved, 1),
            "net_load_mw":     round(net_demand, 1),
            "renewable_pct":   renew_pct,
            "lng_price_usd":   round(lng_price, 2),
            "gas_cost_inr_mwh":round(gas_cost, 0),
            "hourly_cost_inr": cost_hr,
            "carbon_tonnes":   carbon,
            "month":           m,
            "month_name":      ts.strftime("%b"),
            "hour":            h,
            "season":          ("Winter" if m in [12,1,2] else
                                "Summer" if m in [3,4,5,6] else
                                "Monsoon" if m in [7,8,9] else "Autumn"),
            "time_of_day":     ("Night" if h<6 else "Morning" if h<12
                                 else "Afternoon" if h<18 else "Evening"),
            "day_of_week":     ts.day_name(),
        })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} hourly records")
    print(f"  Demand: {df.demand_mw.mean():.0f} avg, {df.demand_mw.max():.0f} peak MW")
    print(f"  Solar:  {df.solar_mw.mean():.0f} avg MW")
    print(f"  Wind:   {df.wind_mw.mean():.0f} avg MW")
    print(f"  Renew%: {df.renewable_pct.mean():.1f}% avg")
    print(f"  Unserved: {df.unserved_mw.mean():.1f} MW avg ({(df.unserved_mw>0).mean()*100:.1f}% hours)")
    return df


def generate_crisis_window(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    India 2024 Summer Heatwave Crisis: May 15 - June 5
    - Peak demand surge (45°C+ across UP, Bihar, Rajasthan)
    - LNG price spike to $18-22/MMBtu
    - Gas plants offline (uneconomical)
    - Solar panel efficiency drops in extreme heat
    """
    crisis = full_df[
        (full_df["datetime"] >= "2024-05-15") &
        (full_df["datetime"] <= "2024-06-05 23:00")
    ].copy().reset_index(drop=True)

    n = len(crisis)
    for i, row in crisis.iterrows():
        # Storm escalates days 5-15, peak around day 10
        progress = i / n
        severity = float(np.sin(max(0, progress - 0.15) * np.pi / 0.70)
                         if 0.15 < progress < 0.85 else 0)

        # Demand spike: extreme heat pushes 30-45% above normal
        crisis.at[i, "demand_mw"] = min(
            DEMAND_PEAK,
            float(row["demand_mw"]) * (1 + 0.32 * severity)
            + float(np.random.normal(0, 2000))
        )
        # LNG price spikes during crisis
        lng_crisis = 14.5 + severity * 8.0   # peaks at ~$22/MMBtu
        gas_plf    = max(0.05, 0.50 - (lng_crisis - 8) * 0.025)
        crisis.at[i, "gas_available_mw"] = round(GAS_INSTALLED * gas_plf, 1)
        crisis.at[i, "lng_price_usd"]    = round(lng_crisis, 2)
        crisis.at[i, "gas_cost_inr_mwh"] = round(gas_cost_per_mwh_custom(lng_crisis), 0)
        # Solar slightly down (efficiency loss at 45°C+)
        crisis.at[i, "solar_mw"] = max(0, float(row["solar_mw"]) * (1 - severity * 0.08))
        # Coal runs flat out (already maxed)

    # Recompute derived fields
    crisis["renewable_mw"]  = crisis["solar_mw"] + crisis["wind_mw"]
    crisis["net_load_mw"]   = (crisis["demand_mw"] - crisis["renewable_mw"]
                                - crisis["coal_mw"]).clip(lower=0)
    crisis["gas_used_mw"]   = crisis[["net_load_mw","gas_available_mw"]].min(axis=1)
    crisis["unserved_mw"]   = (crisis["net_load_mw"] - crisis["gas_used_mw"]).clip(lower=0)
    crisis["renewable_pct"] = (crisis["renewable_mw"] / crisis["demand_mw"] * 100).round(2)

    # EWS signals (calibrated to India grid)
    P75_INDIA = 95_000  # MW net load percentile thresholds
    P90_INDIA = 108_000
    P95_INDIA = 115_000
    crisis["net_load_mw"]     = crisis["net_load_mw"].clip(lower=0)
    crisis["demand_6h_trend"] = crisis["demand_mw"].diff(6).fillna(0)
    crisis["net_load_3h"]     = crisis["net_load_mw"].diff(3).fillna(0)

    def ews_india(row):
        s = 0.0
        nl  = row["net_load_mw"]
        rr  = row["renewable_pct"] / 100
        t6  = row["demand_6h_trend"]
        lng = row["lng_price_usd"]

        # Signal 1: net load pressure
        if nl > P95_INDIA:   s += 0.40
        elif nl > P90_INDIA: s += 0.25
        elif nl > P75_INDIA: s += 0.12

        # Signal 2: gas uneconomical (unique India signal)
        if lng > 18:   s += 0.30
        elif lng > 14: s += 0.15
        elif lng > 11: s += 0.05

        # Signal 3: demand surge
        if t6 > 15_000:  s += 0.20
        elif t6 > 8_000: s += 0.10

        # Signal 4: renewable shortage
        if rr < 0.10:   s += 0.10
        elif rr < 0.15: s += 0.05

        return round(min(s, 1.0), 3)

    crisis["ews_score"]   = crisis.apply(ews_india, axis=1)
    crisis["ews_alert"]   = (crisis["ews_score"] >= 0.45).astype(int)
    crisis["alert_label"] = crisis["ews_alert"].map({0:"Normal", 1:"⚠ ALERT"})
    crisis["load_shedding_hrs"] = ((crisis["unserved_mw"] > 500).astype(int))
    crisis["status"] = np.where(crisis["unserved_mw"] > 2000, "CRISIS",
                       np.where(crisis["ews_alert"] == 1, "WARNING", "NORMAL"))
    crisis["blackout_risk_index"] = (
        (crisis["net_load_mw"] - 70_000) / 50_000
    ).clip(0, 1).round(3)
    crisis["date_label"] = crisis["datetime"].dt.strftime("%d %b")

    print(f"\nIndia Crisis Window: {len(crisis)} hours")
    print(f"  Alerts: {crisis.ews_alert.sum()} hours")
    print(f"  Crisis status: {(crisis.status=='CRISIS').sum()} hours")
    first_alert  = crisis.loc[crisis.ews_alert==1, "datetime"].min()
    first_crisis = crisis.loc[crisis.status=="CRISIS", "datetime"].min()
    if pd.notna(first_alert) and pd.notna(first_crisis):
        lead = (first_crisis - first_alert).total_seconds() / 3600
        print(f"  First alert:  {first_alert}")
        print(f"  First crisis: {first_crisis}")
        print(f"  Lead time:    {lead:.1f} hours")

    return crisis


def gas_cost_per_mwh_custom(lng_price: float) -> float:
    return lng_price * GAS_HEAT_RATE * 83.5


if __name__ == "__main__":
    os.makedirs("data/india", exist_ok=True)
    full_df = generate_full_year()
    full_df.to_csv("data/india/india_hourly_2024.csv", index=False)
    print("Saved: data/india/india_hourly_2024.csv")

    crisis_df = generate_crisis_window(full_df)
    crisis_df.to_csv("data/india/india_crisis_2024.csv", index=False)
    print("Saved: data/india/india_crisis_2024.csv")
