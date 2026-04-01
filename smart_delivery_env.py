import numpy as np
import random

class SmartDeliveryEnv:
    def __init__(
        self,
        num_deliveries=5,
        grid_size=20,
        max_steps=100,
        traffic=True,
        dynamic_orders=True,
        seed=42
    ):
        self.num_deliveries = num_deliveries
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.traffic = traffic
        self.dynamic_orders = dynamic_orders

        random.seed(seed)
        np.random.seed(seed)

        self.reset()

    
    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.time = 0
        self.steps = 0
        self.total_reward = 0

        self.deliveries = []
        self._generate_deliveries(self.num_deliveries)

        return self.state()

    
    def _generate_deliveries(self, n):
        for _ in range(n):
            self.deliveries.append({
                "id": len(self.deliveries),
                "location": np.random.randint(0, self.grid_size, size=2),
                "deadline": np.random.randint(15, 50),
                "priority": random.choice([1, 2, 3]),  # 3 = highest
                "created_at": self.time,
                "done": False
            })

    
    def state(self):
        delivery_state = []

        for d in self.deliveries:
            delivery_state.extend([
                d["location"][0],
                d["location"][1],
                d["deadline"] - self.time,  # remaining time
                d["priority"],
                int(d["done"])
            ])

        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.time,
            *delivery_state
        ], dtype=np.float32)

    
    def step(self, action):
        reward = 0
        done = False
        info = {}

        self.steps += 1

        
        if action < 0 or action >= len(self.deliveries):
            return self.state(), -10, False, {"error": "invalid_action"}

        delivery = self.deliveries[action]

        
        if delivery["done"]:
            return self.state(), -5, False, {"error": "already_done"}

        
        distance = np.linalg.norm(self.agent_pos - delivery["location"])

        
        if self.traffic:
            traffic = np.random.uniform(1.0, 2.5)
            distance *= traffic

        
        self.agent_pos = delivery["location"]
        self.time += distance

        
        time_left = delivery["deadline"] - self.time

        if time_left >= 0:
            reward += 20 * delivery["priority"]   # prioritize important deliveries
        else:
            reward -= 25 * delivery["priority"]   # heavy penalty if late

        # travel cost
        reward -= distance * 0.3

        
        reward += max(0, (10 - abs(time_left)))  

        delivery["done"] = True

        
        if self.dynamic_orders and random.random() < 0.2:
            self._generate_deliveries(1)

        
        if all(d["done"] for d in self.deliveries):
            done = True

        if self.steps >= self.max_steps:
            done = True

        self.total_reward += reward

        
        info = {
            "time": round(self.time, 2),
            "total_reward": round(self.total_reward, 2),
            "pending_orders": sum(not d["done"] for d in self.deliveries)
        }

        return self.state(), reward, done, info



if __name__ == "__main__":
    env = SmartDeliveryEnv()

    state = env.reset()
    done = False

    print("\n Smart Delivery Simulation Started\n")

    while not done:
        
        best_action = None
        best_score = -float("inf")

        for i, d in enumerate(env.deliveries):
            if not d["done"]:
                score = d["priority"] / (
                    np.linalg.norm(env.agent_pos - d["location"]) + 1e-5
                )
                if score > best_score:
                    best_score = score
                    best_action = i

        action = best_action if best_action is not None else 0

        next_state, reward, done, info = env.step(action)

        print(
            f"Step: {env.steps} | Action: {action} | Reward: {round(reward,2)} | "
            f"Time: {info['time']} | Pending: {info['pending_orders']}"
        )

    print("\n Simulation Finished")
    print("Total Reward:", round(env.total_reward, 2))
