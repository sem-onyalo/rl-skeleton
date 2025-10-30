"""A module to manage the creation and usage of Markov Decison Process (MDP) environments."""

import logging
from abc import ABC
from typing import Any

import numpy as np


class MDP(ABC):
    """An abstract class representing an RL Markov Decision Process."""

    def __init__(self) -> None:
        """Instantiates a new `MDP` object."""

        self.logger = logging.getLogger(type(self).__name__)

    @property
    def n_action(self) -> int:
        """The total number of possible actions to take in this MDP."""

        raise RuntimeError("Subclass must implement property.")

    def start(self, *args, **kwargs) -> np.ndarray:
        """
        Starts a new MDP episode.

        Returns:
            (ndarray): The initial state when the episode starts.
        """

        raise RuntimeError("Subclass must implement function.")

    def step(
        self, action: int, *args, **kwargs
    ) -> tuple[float, np.ndarray, bool, dict[str, Any]]:
        """
        Takes a single step in the MDP.

        Args:
            action (int): The index of the action to take.

        Returns:
            (tuple[float, ndarray, bool, dict[str, Any]]):
            Information about the results of the step taken, where, by index:
                0: the reward
                1: the next state
                2: whether or not the next state is a terminal state
                3: any additional information
        """

        raise RuntimeError("Subclass must implement function.")
