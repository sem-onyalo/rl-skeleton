from .agent import Agent
from environment import MDP
from util.constants import *

class Human(Agent):
    """
    This class facilitates a human operating in an MDP.
    """

    def __init__(self, mdp:MDP) -> None:
        super().__init__(HUMAN)
        self.mdp = mdp
        self.mdp.set_operator(HUMAN)
