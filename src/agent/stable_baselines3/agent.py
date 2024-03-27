# from stable_baselines3 import DQN

from ..agent_2 import Agent
from ..common.run_result import RunResult
from environment import Environment

class StableBaselines3(Agent):
    def __init__(self, name:str, env:Environment) -> None:
        super().__init__(name, env)
