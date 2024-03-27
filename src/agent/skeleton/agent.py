from ..agent_2 import Agent
from ..common.run_result import RunResult
# from agent.common.run_result import RunResult
from environment import Environment

class SkeletonAgent(Agent):
    def __init__(self, name:str, env:Environment) -> None:
        super().__init__(name, env)

    def learn(self, max_steps: int) -> RunResult:
        return super().learn(max_steps)
    
    def run(self) -> RunResult:
        return super().run()
