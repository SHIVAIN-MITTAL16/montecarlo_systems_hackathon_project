# src/ai_anomaly.py
# Anomaly Detector using Isolation Forest (100% FREE)
# Detects unusual grid patterns before they become crises

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import os

def train_anomaly_detector(clean_data_path="data/clean_data.csv"):
    """
    Train anomaly detector on normal grid data.
    Think of it as: teaching the AI what 'normal' looks like.
    """
    print("Training anomaly detector...")
    
    df = pd.read_csv(clean_data_path)
    
    # Features to watch (what makes a reading 'normal' or 'suspicious')
    features = ["demand_mw", "solar_mw", "wind_mw"]
    X = df[features].dropna().values
    
    # Train Isolation Forest
    # contamination=0.03 means: expect 3% of readings to be anomalies
    detector = IsolationForest(
        contamination=0.03,
        random_state=42,
        n_estimators=100
    )
    detector.fit(X)
    
    # Save the trained detector
    os.makedirs("data", exist_ok=True)
    with open("data/anomaly_detector.pkl", "wb") as f:
        pickle.dump(detector, f)
    
    print("✅ Anomaly detector trained and saved!")
    return detector


def check_anomaly(demand_mw, solar_mw, wind_mw):
    """
    Check if current grid reading is anomalous.
    
    Returns: (is_anomaly, score, message)
    score: -1.0 to 1.0 
           Below -0.5 = anomaly
           Above 0 = normal
    """
    try:
        with open("data/anomaly_detector.pkl", "rb") as f:
            detector = pickle.load(f)
    except FileNotFoundError:
        return False, 0, "Detector not trained. Run setup.py first."
    
    reading = np.array([[demand_mw, solar_mw, wind_mw]])
    
    # Get anomaly score
    score = detector.score_samples(reading)[0]
    is_anomaly = score < -0.5
    
    if is_anomaly:
        message = (f"⚠️ ANOMALY DETECTED! Score: {score:.3f}\n"
                  f"This grid pattern has never been seen in normal operation.\n"
                  f"Potential crisis forming — check EWS signals immediately.")
    else:
        message = f"✅ Normal operation. Anomaly score: {score:.3f}"
    
    return is_anomaly, score, message


def get_anomaly_explanation(score):
    """Simple explanation of what the score means."""
    if score < -0.8:
        return "🔴 EXTREME anomaly — grid in completely unknown territory"
    elif score < -0.5:
        return "🟡 ANOMALY detected — unusual pattern, monitor closely"
    elif score < -0.2:
        return "🟠 Slightly unusual — worth watching"
    else:
        return "🟢 Normal operation"