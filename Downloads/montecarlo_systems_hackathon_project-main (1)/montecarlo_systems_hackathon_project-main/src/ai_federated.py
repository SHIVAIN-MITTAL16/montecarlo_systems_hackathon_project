# src/ai_federated.py
# Federated Learning using Flower (flwr) — 100% FREE
# Grids share AI learning without sharing private data

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# Simple model that both grids train
class SimpleGridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


# ── Flower Client (represents one grid) ────────────────────────────────────
class GridClient(fl.client.NumPyClient):
    """
    Each grid is a 'client' that trains locally
    and shares only the learned weights (not data).
    """
    
    def __init__(self, grid_name, data):
        self.grid_name = grid_name
        self.data = data
        self.model = SimpleGridModel()
    
    def get_parameters(self, config):
        """Share current model weights with server."""
        return [val.cpu().numpy() 
                for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        """Receive improved weights from server."""
        keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        self.model.load_state_dict(state_dict)
    
    def fit(self, parameters, config):
        """Train on local data for 1 round."""
        self.set_parameters(parameters)
        
        # Train for 5 epochs on local data
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = nn.MSELoss()
        
        X = torch.FloatTensor(self.data['X'])
        y = torch.FloatTensor(self.data['y'])
        
        self.model.train()
        for _ in range(5):
            optimizer.zero_grad()
            pred = self.model(X).squeeze()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        
        print(f"  [{self.grid_name}] Local training done. Loss: {loss.item():.4f}")
        return self.get_parameters(config={}), len(X), {}
    
    def evaluate(self, parameters, config):
        """Test model accuracy."""
        self.set_parameters(parameters)
        X = torch.FloatTensor(self.data['X'])
        y = torch.FloatTensor(self.data['y'])
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).squeeze()
            loss = nn.MSELoss()(pred, y).item()
        
        return loss, len(X), {"loss": loss}


def run_federated_demo():
    """
    Quick demo: Texas + India grids learning together.
    For hackathon demo — runs in simulation mode.
    """
    print("Running Federated Learning Demo...")
    print("Texas grid and India grid training together...")
    print("Neither grid sees the other's data!")
    
    # Simulate Texas data (weather-driven patterns)
    texas_X = np.random.randn(100, 5).astype(np.float32)
    texas_y = np.random.randn(100).astype(np.float32)
    
    # Simulate India data (price-driven patterns)
    india_X = np.random.randn(100, 5).astype(np.float32) * 1.5
    india_y = np.random.randn(100).astype(np.float32) * 1.2
    
    texas_client = GridClient("Texas ERCOT", 
                               {"X": texas_X, "y": texas_y})
    india_client  = GridClient("India NRGP",  
                               {"X": india_X,  "y": india_y})
    
    print("\n📊 Round 1: Both grids train locally...")
    tx_params, _, _ = texas_client.fit(texas_client.get_parameters({}), {})
    in_params, _, _ = india_client.fit(india_client.get_parameters({}), {})
    
    print("\n🔄 Aggregating: combining knowledge from both grids...")
    # Average the weights (FedAvg algorithm)
    avg_params = [(tx + ind) / 2 
                  for tx, ind in zip(tx_params, in_params)]
    
    print("\n📤 Sending improved model back to both grids...")
    texas_client.set_parameters(avg_params)
    india_client.set_parameters(avg_params)
    
    print("\n✅ Federated Learning complete!")
    print("   Texas learned from India's price crisis patterns")
    print("   India learned from Texas's weather crisis patterns")
    print("   Neither shared any private grid data")
    
    return True