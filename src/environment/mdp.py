import logging
from typing import Dict
from typing import Tuple

import numpy as np

class MDP:
    n_state:int
    n_action:int
    d_state:Tuple[int, int]
    operator:str

    def __init__(self, name:str) -> None:
        self.name = name
        self.logger = logging.getLogger(name)

    def start(self) -> np.ndarray:
        """
        Starts a new session for this `MDP`.

        :return The current state.
        """
        return self._get_state()

    def step(self, action:int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        """
        Executes a step using the specified action in this `MDP`.

        :param action: The action to execute.
        :return The next state as a result of the specified action, 
            The reward as a result of the specified action,
            Whether or not the next state is a terminal state,
            Additional info as a result of the specified action
        """
        self._apply_action(action)

        state = self._get_state()

        reward = self._get_reward()

        terminal = self._get_terminal()

        info = self._get_info()

        return state, reward, terminal, info

    def get_operator(self) -> str:
        return self.operator

    def set_operator(self, operator:str) -> None:
        self.operator = operator

    def _apply_action(self, action:np.ndarray) -> None:
        """
        Applies the specified action to this `MDP`.
        """
        raise NotImplementedError()

    def _get_state(self) -> np.ndarray:
        """
        Gets the current state.

        :return The current state.
        """
        raise NotImplementedError()

    def _get_reward(self) -> float:
        """
        Calculates the reward for the current state.

        :return The reward resulting from the supplied action.
        """
        raise NotImplementedError()

    def _get_terminal(self) -> bool:
        """
        Determines whether or not the current state is a terminal state.

        :return Whether or not this `MDP is in a terminal state.
        """
        raise NotImplementedError()

    def _get_info(self) -> Dict[str, object]:
        """
        Returns additional info beyond the current state and reward.

        :return Additional info.
        """
        raise NotImplementedError()
