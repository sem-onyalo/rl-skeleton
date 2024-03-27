from .agent import Agent
from environment import MDP
from skeleton.agent import SkeletonAgent
from util.constants import SB3_DQN
from util.constants import SKL_QLEARNING

class AgentFactory:
    @staticmethod
    def create_agent(id:str, mdp:MDP) -> Agent:
        """
        Creates a new agent with the specified ID.

        :param id: The ID of the agent to create.
        :param mdp: The MDP the agent will run in
        :return The created agent.
        """
        if id == SKL_QLEARNING:
            return SkeletonAgent(id, mdp)
        else:
            raise NotImplementedError(f"agent with ID {id} is not supported")
