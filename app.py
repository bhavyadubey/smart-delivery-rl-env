from fastapi import FastAPI
from env.environment import SmartDeliveryEnv
from inference import run_episode

app = FastAPI()

env = SmartDeliveryEnv()

# Health check endpoint
@app.get("/")
def home():
    return {"status": "running"}

# Reset endpoint (supports BOTH GET and POST)
@app.get("/reset")
@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state.model_dump()}

# Inference endpoint (for testing logs)
@app.get("/run")
def run_inference():
    run_episode()
    return {"status": "inference executed"}
