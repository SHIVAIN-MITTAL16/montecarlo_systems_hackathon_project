# src/ai_forecast.py
# LSTM Demand Forecasting using PyTorch (100% FREE)
# Simple version perfect for beginners

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ── Step 1: Define the LSTM Model ──────────────────────────────────────────
class DemandLSTM(nn.Module):
    """
    LSTM neural network for demand forecasting.
    Think of it as a brain that learns patterns from past data.
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=24):
        super(DemandLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM layer - learns patterns over time
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer - converts learned pattern to forecast
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        prediction = self.linear(lstm_out[:, -1, :])
        return prediction


# ── Step 2: Train the model ─────────────────────────────────────────────────
def train_forecast_model(demand_data, epochs=50):
    """
    Train LSTM on historical demand data.
    
    demand_data = list or array of hourly demand values
    Example: [44000, 45200, 46100, ...]
    """
    print("Training demand forecast model...")
    
    # Normalize data (make all values between 0 and 1)
    demand_array = np.array(demand_data, dtype=np.float32)
    demand_min = demand_array.min()
    demand_max = demand_array.max()
    demand_norm = (demand_array - demand_min) / (demand_max - demand_min)
    
    # Create sequences: use 72 hours to predict next 24 hours
    sequence_length = 72
    X, y = [], []
    
    for i in range(len(demand_norm) - sequence_length - 24):
        X.append(demand_norm[i:i+sequence_length])
        y.append(demand_norm[i+sequence_length:i+sequence_length+24])
    
    X = torch.FloatTensor(X).unsqueeze(-1)  # shape: (samples, 72, 1)
    y = torch.FloatTensor(y)                 # shape: (samples, 24)
    
    # Build and train model
    model = DemandLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")
    
    print("✅ Model trained!")
    
    # Save model and normalization params
    torch.save({
        'model_state': model.state_dict(),
        'demand_min': demand_min,
        'demand_max': demand_max
    }, 'data/demand_forecast_model.pt')
    
    return model, demand_min, demand_max


# ── Step 3: Make predictions ────────────────────────────────────────────────
def forecast_next_24_hours(recent_72_hours):
    """
    Predict next 24 hours of demand.
    
    recent_72_hours = list of last 72 hourly demand readings
    Returns: list of 24 predicted demand values
    """
    try:
        # Load saved model
        checkpoint = torch.load('data/demand_forecast_model.pt')
        demand_min = checkpoint['demand_min']
        demand_max = checkpoint['demand_max']
        
        model = DemandLSTM()
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        # Normalize input
        recent = np.array(recent_72_hours, dtype=np.float32)
        recent_norm = (recent - demand_min) / (demand_max - demand_min)
        
        # Predict
        x = torch.FloatTensor(recent_norm).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_norm = model(x).squeeze().numpy()
        
        # Denormalize back to MW
        pred_mw = pred_norm * (demand_max - demand_min) + demand_min
        
        return pred_mw.tolist()
        
    except FileNotFoundError:
        print("Model not trained yet. Run train_forecast_model() first.")
        return None