# ⚡ Grid Sentinel AI — SIH26061

> **Weather-Aware AI Digital Twin for Resilient Energy Management of Polar Research Stations**

Grid Sentinel AI is a decision-support prototype aligned to **Smart India Hackathon 2026 Problem Statement SIH26061: AI-Driven Smart Energy Management System for Polar Research Stations**.

## What the prototype does

A polar research station is an isolated microgrid. Its critical loads depend on renewable generation, battery storage and backup generation while weather can change both supply and demand.

Grid Sentinel AI models that station as a **Digital Twin** and closes the loop:

**Weather → Forecast → Monte Carlo Simulation → Risk Quantification → Constrained Dispatch → Explainable Recommendation**

### Polar Station Digital Twin

The `/digital-twin` experience is the primary SIH deployment view. It models:

- ☀️ Solar PV
- 🌬️ Wind generation
- 🔋 Battery storage and reserve
- ⛽ Backup generator
- ⚡ Critical/flexible station demand
- ❄️ Polar weather stress

### Weather stress scenarios

Operators can inject nominal weather, a polar storm, a low-light event or wind derating and immediately see the effect on the simulated station.

### Probabilistic risk engine

The browser prototype runs **5,000 deterministic stochastic scenarios per case**. It varies renewable availability, demand and initial battery state, then calculates shortage probability, Expected Unserved Energy (EUE), minimum state of charge and generator usage.

### Optimization / dispatch recommendation

The prototype compares a baseline policy with a reserve-aware dispatch policy. The recommendation prioritizes renewable energy, protects a battery reserve for critical loads and prepares backup generation for residual deficits.

### AI control room

The AI layer is intended to interpret verified simulation/optimization outputs for operators. It should not be treated as the numerical calculation engine.

## Research direction

The project was initially developed as a national-grid resilience prototype. For SIH26061, the primary deployment surface has been changed to the **Polar Research Station Digital Twin**, while the existing national-grid and Texas replay modules remain available as research/reference modules.

This distinction is deliberate: historical grid failures motivate the resilience methodology, but the SIH demo is centered on the isolated polar-station energy problem.

## Technical architecture

```text
Polar Weather / Scenario Inputs
            │
            ▼
   Renewable + Load Model
            │
            ▼
     Digital Twin State
            │
            ▼
  Monte Carlo Risk Engine
       (5,000 futures)
            │
            ├── Shortage Probability
            ├── Expected Unserved Energy
            └── Battery Reserve Risk
            │
            ▼
   Constrained Dispatch Logic
            │
            ▼
 Explainable Operator Recommendation
```

## Technology

- React + TypeScript
- TanStack Router
- Tailwind CSS
- Lucide icons
- Monte Carlo stochastic simulation
- Gemini-based decision-support modules

## Main routes

- `/` — Command Center
- `/digital-twin` — **Polar Research Station Digital Twin (SIH26061 primary demo)**
- `/simulation` — Crisis Lab / stress testing
- `/control-room` — AI Control Room
- `/texas-2021` — historical research replay

## Important validation note

The current Polar Station model is a **research/proof-of-concept simulation**, not a utility-grade EMS/SCADA dispatch controller. Numerical performance claims should be based on reproducible experiments and clearly labeled as simulated results.

## Development

```bash
git clone https://github.com/SHIVAIN-MITTAL16/montecarlo_systems_hackathon_project.git
cd montecarlo_systems_hackathon_project/Downloads/futuregrid-ai-main/futuregrid-ai-main
npm install
npm run dev
```

## SIH alignment

**PS ID:** SIH26061  
**PS:** AI-Driven Smart Energy Management System for Polar Research Stations  
**Theme:** Miscellaneous  
**Category:** Software  
**Team:** MonteCarlo Systems
