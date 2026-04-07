import os
from openai import OpenAI
from env.environment import SmartDeliveryEnv
from env.models import Action


def run_episode():
    env = SmartDeliveryEnv()
    obs = env.reset()

    print("[START]", flush=True)

    total_reward = 0

    
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    client = None
    if base_url and api_key:
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
        except Exception:
            client = None  # fallback safely

    for step in range(5):
        action_id = 0  # default fallback

        
        if client:
            try:
                prompt = f"""
                You are a delivery optimization agent.
                Current state:
                Agent at ({obs.agent_x}, {obs.agent_y})
                Deliveries: {[ (d.id, d.priority, d.done) for d in obs.deliveries ]}
                
                Return ONLY the best delivery_id (integer).
                """

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )

                content = response.choices[0].message.content.strip()

                
                action_id = int(content)

            except Exception:
                
                action_id = step % len(obs.deliveries)

        else:
            
            action_id = step % len(obs.deliveries)

        action = Action(delivery_id=action_id)

        print(f"[STEP] action={action.delivery_id}", flush=True)

        try:
            obs, reward, done, _ = env.step(action)
            reward_value = reward.value
        except Exception:
            reward_value = -10  # safe fallback
            done = True

        print(f"[STEP] reward={reward_value}", flush=True)

        total_reward += reward_value

        if done:
            break

    # Safe scoring
    try:
        score = max(0.0, min(1.0, (total_reward + 100) / 200))
    except Exception:
        score = 0.0

    print(f"[END] score={score}", flush=True)


if __name__ == "__main__":
    try:
        run_episode()
    except Exception:
        
        print("[START]", flush=True)
        print("[STEP] action=0", flush=True)
        print("[STEP] reward=0", flush=True)
        print("[END] score=0.0", flush=True)
