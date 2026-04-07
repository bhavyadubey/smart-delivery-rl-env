import os
from openai import OpenAI
from env.environment import SmartDeliveryEnv
from env.models import Action

# MUST use these
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

def run_episode():
    env = SmartDeliveryEnv()
    obs = env.reset()

    print("[START]", flush=True)

    total_reward = 0

    for step in range(5):
        # Prepare prompt for LLM
        prompt = f"""
        You are a delivery optimization agent.
        Current state:
        Agent at ({obs.agent_x}, {obs.agent_y})
        Deliveries: {[ (d.id, d.x, d.y, d.priority, d.done) for d in obs.deliveries ]}

        Choose the next delivery_id to maximize reward.
        Return ONLY the delivery_id as an integer.
        """

        # REQUIRED API CALL
        response = client.chat.completions.create(
            model=os.environ["MODEL_NAME"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # Extract action
        try:
            action_id = int(response.choices[0].message.content.strip())
        except:
            action_id = 0  # fallback

        action = Action(delivery_id=action_id)

        print(f"[STEP] action={action.delivery_id}", flush=True)

        obs, reward, done, _ = env.step(action)

        print(f"[STEP] reward={reward.value}", flush=True)

        total_reward += reward.value

        if done:
            break

    score = max(0.0, min(1.0, (total_reward + 100) / 200))

    print(f"[END] score={score}", flush=True)


if __name__ == "__main__":
    run_episode()
