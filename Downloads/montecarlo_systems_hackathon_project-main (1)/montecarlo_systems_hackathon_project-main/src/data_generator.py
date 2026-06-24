"""
src/data_generator.py
=====================
Generates realistic ERCOT-Texas-scale synthetic energy data.
Produces: clean_data.csv (8760 hourly rows) + crisis_data.csv (216 rows Feb 10-18 2021)
Uses real statistical profiles from published ERCOT 2021 reports.
"""

import numpy as np
import pandas as pd
import json
import os

np.random.seed(42)

# ── Real ERCOT 2021 statistical profile ──────────────────────────────────────
ERCOT_PROFILE = {
    "demand": {
        "mean": 45994, "std": 7539,
        "min":  28491, "max": 73073,
        "peak_hour": 15,          # 3 PM peak
        "winter_boost": 8000,     # winter demand higher
        "summer_boost": 10000,    # summer AC load
    },
    "solar": {
        "mean": 2257,  "std": 2680,
        "max":  7634,  "night_zero": True,
        "peak_hour": 13,
    },
    "wind": {
        "mean": 7986,  "std": 1504,
        "min":  2090,  "max": 13745,
        "night_boost": True,      # wind stronger at night
    },
}


def _hour_demand_factor(hour: int, month: int) -> float:
    """Realistic hourly demand shape."""
    # Night valley, morning ramp, afternoon peak, evening peak
    base = 0.75 + 0.25 * np.sin((hour - 6) * np.pi / 12)
    if hour < 6:
        base = 0.72 + 0.03 * hour / 6
    elif hour > 20:
        base = 0.85 - 0.15 * (hour - 20) / 4
    # Summer and winter boost
    if month in [6, 7, 8]:
        base *= 1.18
    elif month in [12, 1, 2]:
        base *= 1.12
    return base


def _hour_solar_factor(hour: int) -> float:
    """Solar generation only during daylight hours."""
    if hour < 6 or hour > 20:
        return 0.0
    return max(0, np.sin((hour - 6) * np.pi / 14))


def _hour_wind_factor(hour: int) -> float:
    """Wind stronger at night."""
    return 0.85 + 0.30 * np.cos((hour - 3) * np.pi / 12)


def generate_full_year() -> pd.DataFrame:
    """Generate 8760-hour realistic Texas grid dataset."""
    timestamps = pd.date_range("2021-01-01 00:00", periods=8760, freq="h")
    rows = []

    for ts in timestamps:
        h = ts.hour
        m = ts.month
        doy = ts.day_of_year

        # Seasonal demand base
        seasonal_demand = ERCOT_PROFILE["demand"]["mean"]
        if m in [6, 7, 8]:
            seasonal_demand += ERCOT_PROFILE["demand"]["summer_boost"]
        elif m in [12, 1, 2]:
            seasonal_demand += ERCOT_PROFILE["demand"]["winter_boost"]

        demand = (
            seasonal_demand * _hour_demand_factor(h, m)
            + np.random.normal(0, 1800)
        )
        demand = np.clip(demand,
                         ERCOT_PROFILE["demand"]["min"],
                         ERCOT_PROFILE["demand"]["max"])

        # Solar: daytime only, seasonal
        solar_seasonal = ERCOT_PROFILE["solar"]["mean"] * (
            0.6 + 0.4 * np.sin((doy - 80) * np.pi / 182)
        )
        solar = (
            solar_seasonal * _hour_solar_factor(h) * 2.1
            + np.random.normal(0, 300)
        )
        solar = np.clip(solar, 0, ERCOT_PROFILE["solar"]["max"])

        # Wind: night boost, seasonal
        wind_seasonal = ERCOT_PROFILE["wind"]["mean"] * (
            0.8 + 0.2 * np.cos((doy - 30) * np.pi / 182)
        )
        wind = (
            wind_seasonal * _hour_wind_factor(h)
            + np.random.normal(0, 600)
        )
        wind = np.clip(wind,
                       ERCOT_PROFILE["wind"]["min"],
                       ERCOT_PROFILE["wind"]["max"])

        rows.append({
            "datetime":  ts,
            "demand_mw": round(float(demand), 2),
            "solar_mw":  round(float(solar),  2),
            "wind_mw":   round(float(wind),   2),
        })

    df = pd.DataFrame(rows)
    assert len(df) == 8760
    assert df.isna().sum().sum() == 0
    return df


def generate_crisis_window(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject Winter Storm Uri conditions: Feb 10-18 2021.
    - Demand surges (extreme cold)
    - Wind collapses (frozen turbines)
    - Solar collapses (ice/clouds)
    """
    crisis = full_df[
        (full_df["datetime"] >= "2021-02-10") &
        (full_df["datetime"] <= "2021-02-18 23:00")
    ].copy().reset_index(drop=True)

    # Storm escalation: progressively worse
    n = len(crisis)
    for i, row in crisis.iterrows():
        progress = i / n  # 0 = start, 1 = end
        severity = np.sin(progress * np.pi)  # peaks mid-storm

        # Demand spike: +8000-25000 MW (extreme cold heating load)
        crisis.at[i, "demand_mw"] = min(
            73073,
            row["demand_mw"] + 8000 + severity * 22000
            + np.random.normal(0, 1500)
        )
        # Wind collapse: 80% reduction at peak
        crisis.at[i, "wind_mw"] = max(
            0, row["wind_mw"] * (1 - severity * 0.85)
            + np.random.normal(0, 100)
        )
        # Solar collapse: 90% reduction (ice on panels)
        crisis.at[i, "solar_mw"] = max(
            0, row["solar_mw"] * (1 - severity * 0.90)
        )

    crisis = crisis.reset_index(drop=True)

    # Compute early warning features
    crisis["renewable"] = crisis["solar_mw"] + crisis["wind_mw"]
    crisis["net_load"]  = crisis["demand_mw"] - crisis["renewable"]
    crisis["renewable_ratio"] = crisis["renewable"] / crisis["demand_mw"]
    crisis["demand_6h_trend"]   = crisis["demand_mw"].diff(6).fillna(0)
    crisis["net_load_3h_trend"] = crisis["net_load"].diff(3).fillna(0)

    # Early warning score
    P95 = 48163
    P90 = 46459
    P75 = 43073

    def ews(row):
        s = 0.0
        if row["net_load"] > P95:   s += 0.45
        elif row["net_load"] > P90: s += 0.30
        elif row["net_load"] > P75: s += 0.15
        if row["renewable_ratio"] < 0.12:  s += 0.25
        elif row["renewable_ratio"] < 0.16: s += 0.12
        if row["demand_6h_trend"] > 10000:  s += 0.20
        elif row["demand_6h_trend"] > 6000: s += 0.10
        if row["net_load_3h_trend"] > 6000: s += 0.10
        return min(s, 1.0)

    crisis["ews_score"] = crisis.apply(ews, axis=1)
    crisis["ews_alert"] = (crisis["ews_score"] >= 0.45).astype(int)

    # Blackout risk index (derived from net load pressure)
    crisis["blackout_risk_index"] = (
        (crisis["net_load"] - 40000) / 35000
    ).clip(0, 1) * 0.95

    return crisis


def compute_stats(df: pd.DataFrame, save_path: str = "stats.json") -> dict:
    """Compute and save statistical profile."""
    stats = {}
    for col in ["demand_mw", "solar_mw", "wind_mw"]:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std":  float(df[col].std()),
            "min":  float(df[col].min()),
            "max":  float(df[col].max()),
            "p5":   float(df[col].quantile(0.05)),
            "p25":  float(df[col].quantile(0.25)),
            "p50":  float(df[col].quantile(0.50)),
            "p75":  float(df[col].quantile(0.75)),
            "p95":  float(df[col].quantile(0.95)),
        }
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def run_pipeline(output_dir: str = "data") -> dict:
    """Full data pipeline: generate → save → return paths."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating full-year ERCOT grid data...")
    full_df = generate_full_year()

    clean_path = os.path.join(output_dir, "clean_data.csv")
    full_df.to_csv(clean_path, index=False)
    print(f"  clean_data.csv → {len(full_df)} rows")

    print("Injecting Winter Storm Uri crisis conditions...")
    crisis_df = generate_crisis_window(full_df)
    crisis_path = os.path.join(output_dir, "crisis_data.csv")
    crisis_df.to_csv(crisis_path, index=False)
    print(f"  crisis_data.csv → {len(crisis_df)} rows")

    print("Computing statistics...")
    stats = compute_stats(full_df, os.path.join(output_dir, "stats.json"))
    print(f"  stats.json saved")

    print("Data pipeline complete.")
    return {"full_df": full_df, "crisis_df": crisis_df, "stats": stats}


if __name__ == "__main__":
    run_pipeline()
