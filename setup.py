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

print("\n" + "=" * 55)
print("  Setup complete! All data files generated.")
print("  Now run:  streamlit run app.py")
print("=" * 55)
