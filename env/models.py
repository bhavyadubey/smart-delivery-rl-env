from pydantic import BaseModel
from typing import List

class Delivery(BaseModel):
    id: int
    x: int
    y: int
    deadline: float
    priority: int
    done: bool = False

class Observation(BaseModel):
    agent_x: int
    agent_y: int
    time: float
    deliveries: List[Delivery]

class Action(BaseModel):
    delivery_id: int

class Reward(BaseModel):
    value: float
