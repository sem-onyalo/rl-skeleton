import logging
from typing import List

from common.run_result import RunResult
from environment import Environment

class Agent:
    def __init__(self, name:str, env:Environment) -> None:
        self.name = name
        self.env = env
        self.logger = logging.getLogger(self.name)

    def learn(self, max_steps:int) -> RunResult:
        """
        Run a learning session for this agent.

        :param max_steps: The maximum number of steps this agent should run for.
        :return The result of the run.
        """
        raise NotImplementedError()
    
    def run(self) -> RunResult:
        """
        Run a deterministic policy session for this agent.
        
        :return The result of the run.
        """
        raise NotImplementedError()
