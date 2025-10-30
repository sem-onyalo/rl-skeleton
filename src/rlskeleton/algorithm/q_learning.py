"""A module to manage the creation and usage of the Q-Learning algorithm."""

from typing import Any, cast

import numpy as np

from rlskeleton.function import Function
from rlskeleton.mdp import MDP

from .algorithm import Algorithm, AlgorithmParams


class QLearningParams(AlgorithmParams):
    change_rate: float
    discount_rate: float


class QLearning(Algorithm):
    """Represents the Q-Learning off-policy control algorithm."""

    def __init__(self, mdp: MDP, function: Function, params: QLearningParams) -> None:
        """
        Instantiates a new `QLearning` object.

        Args:
            mdp (MDP): The MDP this algorithm will interact with.
            function (Function): The policy function this algorithm will use.
            params (QLearningParams): The configuration parameters for this algorithm.
        """

        super().__init__(mdp, function, params)

    @property
    def config_params(self) -> QLearningParams:
        """Configuration parameters for this algorithm."""

        return cast(QLearningParams, self._params)

    def _end_step(
        self,
        episode: int,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        """
        Updates the action-value function.

        Args:
            state (ndarray): The current state.
            action (int): The action taken at the current state.
            next_state (ndarray): The resulting state from the action taken.
            reward (float): The reward resulting from the action taken.
        """

        value = self.function.get_value(state, action)

        max_value = self.function.get_value(next_state, self.function(next_state))

        # Q(S,A) = Q(S,A) + a * (R + y * max_a[Q(S',a)] - Q(S,A))
        new_value = value + self._params.change_rate * (  # type: ignore
            reward + (self._params.discount_rate * max_value) - value  # type: ignore
        )

        self.function.update(state, action, new_value)

        super()._end_step(episode, state, action, next_state, reward, info)

    @staticmethod
    def parse_config_params(obj: Any) -> QLearningParams:
        """
        Parses an object to get the configuration parameters for this class.

        Args:
            obj (Any): The object to parse. The object must implement `__getitem__`.

        Returns:
            The configuration parameters for this class.
        """

        decay_delay = (
            tuple(obj["decay_delay"].split("=")) if obj["decay_delay"] else None
        )

        return QLearningParams(
            change_rate=obj["change_rate"],
            discount_rate=obj["discount_rate"],
            # TODO: Modify this so that parent attributes (below) are set in parent,
            # otherwise, we'll need to duplicate this for all `Algorithm` child classes :(
            decay_delay=decay_delay,
            episodes=obj["episodes"],
        )
