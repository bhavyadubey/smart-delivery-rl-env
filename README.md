# Smart Delivery RL Environment

A real-world inspired reinforcement learning environment simulating last-mile delivery optimization with dynamic demand, stochastic delays, and priority-based decision making.

## Features
- Custom RL environment with `reset()`, `step()`, `state()` API
- Dynamic order generation (real-time simulation)
- Stochastic traffic delays
- Priority-based delivery system
- Reward shaping for intelligent decision-making
- Greedy baseline agent for demonstration

## Use Case
This environment models real-world logistics problems such as:
- Last-mile delivery optimization
- Fleet routing under uncertainty
- Time-sensitive task scheduling



##  Run the Simulation

```bash
python smart_delivery_env.py

```
## 📊 Sample Output

Step: 1 | Action: 2 | Reward: 18.4 | Time: 3.2 | Pending: 4
Step: 2 | Action: 1 | Reward: 12.7 | Time: 6.5 | Pending: 3
Step: 3 | Action: 0 | Reward: -5.2 | Time: 9.8 | Pending: 3
Step: 4 | Action: 3 | Reward: 21.1 | Time: 13.4 | Pending: 2
...
