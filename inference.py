import os
from openai import OpenAI
from env.environment import SmartDeliveryEnv
from env.models import Action

def run_episode():
    env = SmartDeliveryEnv()
    obs = env.reset()

    print("[START]", flush=True)

    total_reward = 0

    # Safely get env vars
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    client = None
    if base_url and api_key:
        client = OpenAI(base_url=base_url, api_key=api_key)

    for step in range(5):
        if client:
            # LLM call (ONLY in hackathon environment)
            prompt = f"""
            Agent at ({obs.agent_x}, {obs.agent_y})
            Deliveries: {[ (d.id, d.priority, d.done) for d in obs.deliveries ]}
            Return best delivery_id.
            """

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            try:
                action_id = int(response.choices[0].message.content.strip())
            except:
                action_id = 0
        else:
            # fallback for HF / local
            action_id = step % len(obs.deliveries)

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
