# src/ai_chatbot.py
# GridAI Chatbot using Ollama (100% FREE, runs on your laptop)

import requests
import json

def ask_grid_ai(question, grid_data):
    """
    Ask the AI a question about the grid.
    
    grid_data = dictionary with current grid status
    Example: {"ews_score": 0.67, "demand_mw": 58000, 
               "risk": "HIGH", "gas_price": 80}
    """
    
    # Build context from current grid data
    context = f"""
    You are GridAI, an expert power grid analyst.
    
    Current grid status right now:
    - EWS (danger) score: {grid_data.get('ews_score', 0)} out of 1.0
    - Electricity demand: {grid_data.get('demand_mw', 0)} MW
    - Risk level: {grid_data.get('risk', 'UNKNOWN')}
    - Gas price: ${grid_data.get('gas_price', 80)} per MWh
    - Blackout risk: {grid_data.get('blackout_risk', 0)}%
    
    Answer the question clearly in 3-5 sentences.
    Give specific numbers and actionable advice.
    """
    
    # Call Ollama (running locally on your laptop)
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1",
                "prompt": context + "\n\nQuestion: " + question,
                "stream": False,
                "options": {"num_predict": 200}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return "AI is loading... Please try again in 10 seconds."
            
    except Exception as e:
        return f"Make sure Ollama is running: open terminal and type 'ollama serve'"


def check_ollama_running():
    """Check if Ollama is running on this computer."""
    try:
        r = requests.get("http://localhost:11434", timeout=2)
        return True
    except:
        return False