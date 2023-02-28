import io
import logging

import numpy as np

from mdp import MDP

class Policy:
    epsilon:float
    decay_type:str
    model_file_ext:str

    def __init__(self, mdp:MDP, name:str) -> None:
        self.mdp = mdp
        self.name = name
        self.logger = logging.getLogger(self.name)

    def __call__(self, state:object) -> int:
        """
        Return the optimal (max) action.
        """
        pass

    def choose_action(self, state:str) -> int:
        """
        Choose an action stochastically.
        """
        pass

    def get_epsilon_greedy_action(self, state: str) -> int:
        """
        Choose an action using the epsilon-greedy algorithm.
        """
        pass

    def get_value(self, state:str, action:int) -> float:
        """
        Returns the value for a state and action.
        """
        pass

    def get_values(self, state:str) -> np.ndarray:
        """
        Returns the values for each action.
        """
        pass

    def decay(self, value:float) -> None:
        """
        Decay the epsilon value according to the current exploration/exploitation strategy.
        """
        pass

    def transform_state(self, state:object) -> str:
        """
        Transform a state to the format the action-value function uses.
        """
        pass

    def update(self, *args) -> None:
        """
        Updates the action-value function.
        """
        pass

    def get_model(self) -> io.BytesIO:
        """
        Gets the model parameters as a bytes-like object.
        """
        pass

    def load_model(self, buffer:io.BytesIO) -> None:
        """
        Loads the model parameters from a bytes-like object.
        """
        pass
