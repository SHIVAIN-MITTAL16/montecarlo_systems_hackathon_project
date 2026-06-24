# src/ai_battery_rl.py
# Reinforcement Learning Battery Dispatcher (100% FREE)
# AI learns optimal battery strategy by trial and error

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os

# ── Step 1: Create the Grid Environment ────────────────────────────────────
class GridBatteryEnv(gym.Env):
    """
    A simulation environment where the RL agent learns
    to manage the battery. Think of it as a video game
    where the agent plays millions of rounds to get better.
    """
    
    def __init__(self):
        super().__init__()
        
        # Battery settings
        self.battery_max_mw  = 2000    # max power output
        self.battery_max_mwh = 8000    # max energy stored
        self.battery_rte     = 0.90    # 90% efficiency
        
        # Action: how much to charge (+) or discharge (-)
        # Range: -1.0 (full discharge) to +1.0 (full charge)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # State: what the agent can observe
        # [demand, solar, wind, battery_level, ews_score, hour_of_day]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        """Start a new episode."""
        self.soc = 0.5 * self.battery_max_mwh  # start at 50%
        self.step_count = 0
        
        # Random starting grid state
        self.demand  = np.random.uniform(30000, 70000)
        self.solar   = np.random.uniform(0, 5000)
        self.wind    = np.random.uniform(2000, 10000)
        self.ews     = np.random.uniform(0, 0.8)
        self.hour    = np.random.randint(0, 24)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Convert grid state to numbers between 0 and 1."""
        return np.array([
            self.demand  / 72000,              # normalize demand
            self.solar   / 5500,               # normalize solar
            self.wind    / 11000,              # normalize wind
            self.soc     / self.battery_max_mwh, # battery level
            self.ews,                           # EWS score already 0-1
            self.hour    / 24                   # hour of day
        ], dtype=np.float32)
    
    def step(self, action):
        """
        Agent takes an action. We calculate the reward.
        Good action = low cost + high reliability = high reward.
        """
        # Convert action to MW
        action_mw = float(action[0]) * self.battery_max_mw
        
        # Apply battery action
        if action_mw > 0:  # charging
            charge = min(action_mw, 
                        (self.battery_max_mwh - self.soc) / self.battery_rte)
            self.soc += charge * self.battery_rte
            bat_power = -charge  # consuming power to charge
        else:  # discharging
            discharge = min(-action_mw, self.soc, self.battery_max_mw)
            self.soc -= discharge
            bat_power = discharge  # providing power
        
        self.soc = np.clip(self.soc, 
                          0.05 * self.battery_max_mwh,
                          0.95 * self.battery_max_mwh)
        
        # Calculate how much gas is needed
        renewable = self.solar + self.wind
        gap = self.demand - renewable - bat_power
        gas_used = max(0, min(gap, self.demand * 0.98))
        unserved = max(0, gap - gas_used)
        
        # Calculate cost
        cost = gas_used * 80 + unserved * 5000
        
        # Reward: negative cost (we want to minimize cost)
        reward = -cost / 1e6  # scale down
        
        # Extra bonus for avoiding blackouts
        if unserved == 0:
            reward += 1.0
        
        # Extra penalty during high EWS score
        if self.ews > 0.45 and unserved > 0:
            reward -= 2.0
        
        # Update grid state for next step
        self.step_count += 1
        self.hour = (self.hour + 1) % 24
        
        # Random grid variations
        self.demand  += np.random.normal(0, 500)
        self.demand   = np.clip(self.demand, 28000, 72000)
        self.solar   += np.random.normal(0, 200)
        self.solar    = np.clip(self.solar, 0, 5500)
        self.wind    += np.random.normal(0, 400)
        self.wind     = np.clip(self.wind, 0, 11000)
        
        done = self.step_count >= 8760  # one full year
        
        return self._get_observation(), reward, done, False, {}


# ── Step 2: Train the RL Agent ──────────────────────────────────────────────
def train_rl_agent(total_steps=100_000):
    """
    Train the RL agent. This may take 5-15 minutes.
    The agent plays the grid game 100,000 times to learn.
    """
    print(f"Training RL agent for {total_steps:,} steps...")
    print("This takes about 5-10 minutes. Go grab a coffee ☕")
    
    env = GridBatteryEnv()
    
    # PPO is the best RL algorithm for this type of problem
    agent = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1
    )
    
    agent.learn(total_timesteps=total_steps)
    
    os.makedirs("data", exist_ok=True)
    agent.save("data/rl_battery_agent")
    print("✅ RL agent trained and saved!")
    
    return agent


# ── Step 3: Use the trained agent ──────────────────────────────────────────
def get_rl_battery_action(demand_mw, solar_mw, wind_mw, 
                           battery_soc, ews_score, hour):
    """
    Get AI's recommended battery action.
    
    Returns: action_mw (positive = charge, negative = discharge)
    """
    try:
        agent = PPO.load("data/rl_battery_agent")
        
        obs = np.array([
            demand_mw  / 72000,
            solar_mw   / 5500,
            wind_mw    / 11000,
            battery_soc / 8000,
            ews_score,
            hour / 24
        ], dtype=np.float32)
        
        action, _ = agent.predict(obs, deterministic=True)
        action_mw = float(action[0]) * 2000
        
        if action_mw > 0:
            return action_mw, f"⚡ Charge battery: {action_mw:.0f} MW"
        else:
            return action_mw, f"🔋 Discharge battery: {-action_mw:.0f} MW"
            
    except FileNotFoundError:
        return 0, "RL agent not trained yet. Run train_rl_agent() first."