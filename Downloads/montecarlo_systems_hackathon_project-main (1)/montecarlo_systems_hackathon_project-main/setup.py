"""
setup.py
========
Run this ONCE before launching the dashboard.
Generates all required data files.

Usage:
    python setup.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 55)
print("  MonteCarlo Systems — Setup")
print("  Renewable Energy Decision Engine")
print("=" * 55)

os.makedirs("data", exist_ok=True)

# Step 1: Generate clean data
print("\n[1/3] Generating ERCOT grid data...")
from src.data_generator import run_pipeline
run_pipeline(output_dir="data")

# Step 2: Monte Carlo simulation
print("\n[2/3] Running Monte Carlo simulation (5000 scenarios)...")
from src.monte_carlo import run_monte_carlo
run_monte_carlo(n=5_000, output_dir="data")

# Step 3: Optimizer input
print("\n[3/3] Preparing optimizer input...")
from src.optimizer import prepare_optimizer_input
prepare_optimizer_input(
    scenarios_path="data/scenarios.csv",
    output_path="data/optimizer_input.csv"
)

# ── AI Model Training ──────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING AI MODELS...")
print("=" * 55)

import pandas as pd
df = pd.read_csv("data/clean_data.csv")

# AI Step 1: Train anomaly detector
print("\n[AI 1/3] Training anomaly detector...")
from src.ai_anomaly import train_anomaly_detector
train_anomaly_detector()
print("Anomaly detector ready!")

# AI Step 2: Train demand forecast model
print("\n[AI 2/3] Training demand forecast model...")
from src.ai_forecast import train_forecast_model
train_forecast_model(df["demand_mw"].tolist(), epochs=50)
print("AI forecast model ready!")

# AI Step 3: Train RL battery agent (optional — takes 5–10 mins)
print("\n[AI 3/3] Training RL battery agent...")
print("This takes 5-10 minutes. Skip with Ctrl+C if short on time.")
try:
    from src.ai_battery_rl import train_rl_agent
    train_rl_agent(total_steps=50_000)
    print("RL battery agent ready!")
except KeyboardInterrupt:
    print("RL training skipped.")

# ── Done ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅ ALL AI MODELS READY!")
print("  Setup complete! All data files generated.")
print("  Now run:  streamlit run app.py")
print("=" * 55)