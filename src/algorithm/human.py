from .algorithm import Algorithm
from mdp import MDP
from util.constants import *

class Human(Algorithm):
    """
    This class facilitates a human operating in an MDP.
    """

    def __init__(self, mdp:MDP) -> None:
        super().__init__(HUMAN)
        self.mdp = mdp
        self.mdp.set_operator(HUMAN)
