import os
from env.environment import SmartDeliveryEnv
from env.models import Action

print("[START]")

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy")

env = SmartDeliveryEnv()
obs = env.reset()

done = False
step_id = 0
total_reward = 0

while not done:
    best = None
    best_score = -1

    for d in obs.deliveries:
        if not d.done:
            dist = ((obs.agent_x - d.x)**2 + (obs.agent_y - d.y)**2)**0.5
            score = d.priority / (dist + 1e-5)
            if score > best_score:
                best_score = score
                best = d

    action = Action(delivery_id=best.id)

    obs, reward, done, _ = env.step(action)
    total_reward += reward.value

    print(f"[STEP] step={step_id} reward={reward.value}")
    step_id += 1

print(f"[END] total_reward={total_reward}")
