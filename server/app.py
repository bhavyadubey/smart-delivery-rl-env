from fastapi import FastAPI
import uvicorn
from env.environment import SmartDeliveryEnv
from inference import run_episode

app = FastAPI()
env = SmartDeliveryEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/reset")
@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state.model_dump()}

@app.get("/run")
def run_inference():
    run_episode()
    return {"status": "inference executed"}


# REQUIRED by validator
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# REQUIRED for execution
if __name__ == "__main__":
    main()
