import numpy as np
import random
from .models import Observation, Action, Reward, Delivery

class SmartDeliveryEnv:

    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.reset()

    def reset(self):
        self.agent = np.array([0, 0])
        self.time = 0
        self.steps = 0

        self.deliveries = [
            Delivery(
                id=i,
                x=int(np.random.randint(0, 20)),
                y=int(np.random.randint(0, 20)),
                deadline=float(np.random.randint(20, 60)),
                priority=int(random.choice([1, 2, 3]))
            )
            for i in range(5)
        ]

        return self.state()

    def state(self):
        return Observation(
            agent_x=int(self.agent[0]),
            agent_y=int(self.agent[1]),
            time=self.time,
            deliveries=self.deliveries
        )

    def step(self, action: Action):
        self.steps += 1

        delivery = next((d for d in self.deliveries if d.id == action.delivery_id), None)

        if delivery is None or delivery.done:
            return self.state(), Reward(value=-10), False, {"error": "invalid"}

        distance = np.linalg.norm(self.agent - np.array([delivery.x, delivery.y]))
        traffic = np.random.uniform(1.0, 2.0)
        distance *= traffic

        self.agent = np.array([delivery.x, delivery.y])
        self.time += distance

        # reward shaping
        time_left = delivery.deadline - self.time

        reward = 0
        if time_left >= 0:
            reward += 10 * delivery.priority
        else:
            reward -= 15 * delivery.priority

        reward -= distance * 0.2
        reward += max(0, 5 - abs(time_left))  # partial progress signal

        delivery.done = True

        done = all(d.done for d in self.deliveries) or self.steps > 50

        return self.state(), Reward(value=reward), done, {}
