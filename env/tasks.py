from .environment import SmartDeliveryEnv
from .models import Action



def task_easy():
    env = SmartDeliveryEnv()
    obs = env.reset()

    total_reward = 0

    for _ in range(10):
        action = Action(delivery_id=0)
        obs, reward, done, _ = env.step(action)
        total_reward += reward.value
        if done:
            break

    return max(0.0, min(1.0, total_reward / 50))



def task_medium():
    env = SmartDeliveryEnv()
    obs = env.reset()

    total_reward = 0

    for _ in range(20):
        best = max(
            [d for d in obs.deliveries if not d.done],
            key=lambda d: d.priority,
            default=None
        )

        if best is None:
            break

        action = Action(delivery_id=best.id)
        obs, reward, done, _ = env.step(action)
        total_reward += reward.value

        if done:
            break

    return max(0.0, min(1.0, total_reward / 100))



def task_hard():
    env = SmartDeliveryEnv()
    obs = env.reset()

    total_reward = 0

    for _ in range(30):
        best = None
        best_score = -1

        for d in obs.deliveries:
            if not d.done:
                dist = ((obs.agent_x - d.x)**2 + (obs.agent_y - d.y)**2)**0.5
                score = d.priority / (dist + 1e-5)

                if score > best_score:
                    best_score = score
                    best = d

        if best is None:
            break

        action = Action(delivery_id=best.id)
        obs, reward, done, _ = env.step(action)
        total_reward += reward.value

        if done:
            break

    return max(0.0, min(1.0, total_reward / 150))



TASKS = {
    "easy": task_easy,
    "medium": task_medium,
    "hard": task_hard,
}
