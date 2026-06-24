# src/ai_report.py
# Auto Crisis Report Generator using Mistral (100% FREE)

import requests
from datetime import datetime

def generate_crisis_report(grid_data, scenarios_data=None):
    """
    Generate a full crisis report automatically.
    
    grid_data = current grid status dictionary
    Returns: formatted report as string
    """
    
    timestamp = datetime.now().strftime("%d %B %Y, %H:%M")
    
    prompt = f"""
    You are a professional power grid analyst.
    Write a crisis report with these exact sections:
    
    GRID STATUS REPORT
    Generated: {timestamp}
    
    1. EXECUTIVE SUMMARY (2 sentences)
    2. CURRENT SITUATION (3 bullet points)
    3. IMMEDIATE ACTIONS REQUIRED (numbered list of 4 actions)
    4. RISK FORECAST NEXT 24 HOURS (3 time periods)
    5. COMPARISON TO HISTORICAL CRISES (1 sentence)
    
    Grid data to use:
    - EWS Score: {grid_data.get('ews_score', 0)}
    - Demand: {grid_data.get('demand_mw', 0)} MW
    - Risk Level: {grid_data.get('risk', 'UNKNOWN')}
    - Blackout Risk: {grid_data.get('blackout_risk', 0)}%
    - Gas Price: ${grid_data.get('gas_price', 80)}/MWh
    
    Be specific with numbers. Be professional.
    """
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 400}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return "Report generation failed. Check Ollama is running."
            
    except Exception as e:
        return f"Error: {str(e)}. Run 'ollama serve' in terminal."