from fastapi import FastAPI
from env.environment import SmartDeliveryEnv

app = FastAPI()

env = SmartDeliveryEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/reset")
def reset():
    state = env.reset()
    return {"state": state.dict()}
