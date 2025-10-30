"""A module to manage the creation and usage of functions and policies."""

import json
import logging
import math
import random
from abc import ABC
from enum import Enum
from io import BytesIO
from typing import Any

import numpy as np
from pydantic import BaseModel

from .mdp import MDP


class DecayType(Enum):
    EXPONENTIAL = "exponential"
    GLIE = "glie"


class ExploreType(Enum):
    EPSILON_GREEDY = "epsilon-greedy"
    UCB_EXPLORE = "ucb-explore"


class FunctionParams(BaseModel):
    """Configuration parameters for a `Function` object."""

    decay_rate: float
    decay_type: DecayType
    epsilon_floor: float
    epsilon_start: float
    explore_type: ExploreType


class Function(ABC):
    """An abstract class representing an RL policy function."""

    def __init__(self, mdp: MDP, params: FunctionParams) -> None:
        """
        Instantiates a new `Function` object.

        Args:
            mdp (MDP): The MDP this function will use.
            params (FunctionParams): The configuration parameters for this function.
        """

        self.logger = logging.getLogger(type(self).__name__)

        self.mdp = mdp

        self._params = params

        self._epsilon = params.epsilon_start

        self.logger.debug(f"epsilon: {self._epsilon}")

    def __call__(self, state: np.ndarray) -> int:
        """
        Returns the optimal action from the specified state.

        Args:
            state (ndarray): The state from which to pick the optimal action.

        Returns:
            (int): The optimal action.
        """

        return int(np.argmax(self.get_values(state)))

    @property
    def epsilon(self) -> float:
        """The current epsilon value."""

        return self._epsilon

    @property
    def model_file_ext(self) -> str:
        """The file extenstion of the model this function creates."""

        raise RuntimeError("Subclass must implement property.")

    @property
    def config_params(self) -> FunctionParams:
        """Configuration parameters for this function."""

        return self._params

    def choose_action(self, state: np.ndarray) -> int:
        """
        Chooses an action stochastically.

        Args:
            state (ndarray): The state from which to choose the action.

        Returns:
            (int): The chosen action.
        """

        raise RuntimeError("Subclass must implement function.")

    def decay(self, value: float) -> None:
        """
        Decay the current epsilon value according to the set exploration/exploitation strategy.

        Args:
            value (float): The value used to decay epsilon.
        """

        if (
            self._params.decay_type == DecayType.EXPONENTIAL
            and self._params.decay_rate > 0
        ):
            self._epsilon = self._params.epsilon_floor + (
                self._epsilon - self._params.epsilon_floor
            ) * math.exp(-1.0 * value / self._params.decay_rate)

        elif self._params.decay_type == DecayType.GLIE:
            self._epsilon = 1 / value

    def get_model(self) -> BytesIO:
        """
        Retrieves the model parameters as a bytes-like object.

        Returns:
            (BytesIO): The model parameters.
        """

        raise RuntimeError("Subclass must implement function.")

    def get_value(self, state: np.ndarray, action: int) -> float:
        """
        Returns the value for the specifed state and action.

        Args:
            state (ndarray): The state from which to retrieve the action value.
            action (int): The action.

        Returns:
            (float): The value for the action at the given state.
        """

        raise RuntimeError("Subclass must implement function.")

    def get_values(self, state: np.ndarray) -> np.ndarray:
        """
        Returns the values for each action from the specified state.

        Args:
            state (ndarray): The state from which to retrieve the action values.

        Returns:
            (ndarray): The values for each possible action.
        """

        raise RuntimeError("Subclass must implement function.")

    def load_model(self, buffer: BytesIO) -> None:
        """
        Loads model parameters from a bytes-like object.

        Args:
            buffer (BytesIO): The buffer containing the model parameters.
        """

        raise RuntimeError("Subclass must implement function.")

    def update(self, *args) -> None:
        """Updates the action-value function."""

        raise RuntimeError("Subclass must implement function.")

    @staticmethod
    def parse_config_params(obj: Any) -> FunctionParams:
        """
        Parses an object to get the configuration parameters for this class.

        Args:
            obj (Any): The object to parse. The object must implement `__getitem__`.

        Returns:
            (FunctionParams): The configuration parameters for this class.
        """

        return FunctionParams(
            decay_rate=obj["decay_rate"],
            decay_type=obj["decay_type"],
            epsilon_floor=obj["epsilon_floor"],
            epsilon_start=obj["epsilon_start"],
            explore_type=obj["explore_type"],
        )


class TabularPolicy(Function):
    """A class representing an RL tabular policy function."""

    def __init__(self, mdp: MDP, params: FunctionParams) -> None:
        """
        Instantiates a new `TabularPolicy` object.

        Args:
            mdp (MDP): The MDP this function will use.
            params (FunctionParams): The configuration parameters for this function.
        """

        super().__init__(mdp, params)

        self._init_value = 0.0
        self._model_file_ext = ".json"
        self._value_map = {}

    @property
    def model_file_ext(self) -> str:
        """The file extenstion of the model this policy creates."""

        return self._model_file_ext

    def choose_action(self, state: np.ndarray) -> int:
        """
        Chooses an action stochastically.

        Args:
            state (ndarray): The state from which to choose the action.

        Returns:
            (int): The chosen action.
        """

        if self._params.explore_type == ExploreType.EPSILON_GREEDY:
            return self.choose_action_epsilon_greedy(state)
        else:
            raise RuntimeError(
                f"{self._params.explore_type} invalid or not yet implemented"
            )

    def choose_action_epsilon_greedy(self, state: np.ndarray) -> int:
        """
        Chooses an action using the epsilon-greedy algorithm.

        Args:
            state (ndarray): The state from which to choose the action.

        Returns:
            (int): The chosen action.
        """

        do_explore = random.random() < self._epsilon

        self.logger.debug(f"do_explore: {do_explore}")

        return (
            random.randrange(self.mdp.n_action) if do_explore else self.__call__(state)
        )

    def get_model(self) -> BytesIO:
        """
        Retrieves the model parameters as a bytes-like object.

        Returns:
            (BytesIO): The model parameters.
        """

        buffer = BytesIO()

        state_dict = {state: list(self._value_map[state]) for state in self._value_map}

        buffer.write(json.dumps(state_dict, indent=4).encode("utf-8"))

        return buffer

    def get_value(self, state: np.ndarray, action: int) -> float:
        """
        Returns the value for the specifed state and action.

        Args:
            state (ndarray): The state from which to retrieve the action value.
            action (int): The action.

        Returns:
            (float): The value for the action at the given state.
        """

        return self.get_values(state)[action]

    def get_values(self, state: np.ndarray) -> np.ndarray:
        """
        Returns the values for each action from the specified state.

        Args:
            state (ndarray): The state from which to retrieve the action values.

        Returns:
            (ndarray): The values for each possible action.
        """

        state_t = self._lazy_load(state)

        return self._value_map[state_t]

    def load_model(self, buffer: BytesIO) -> None:
        """
        Loads model parameters from a bytes-like object.

        Args:
            buffer (BytesIO): The buffer containing the model parameters.
        """

        state_dict = json.loads(buffer.getvalue())

        for state in state_dict:
            self._value_map[state] = np.asarray(state_dict[state])

    def update(self, state: np.ndarray, action: int, value: float) -> None:
        """
        Updates the action-value function.

        Args:
            state (ndarray): The state to update.
            action (int): The action to update.
            value (float): The new value to assign to the given state and action.
        """

        state_t = self._lazy_load(state)

        self._value_map[state_t][action] = value

    def _lazy_load(self, state: np.ndarray) -> str:
        """
        Loads the values for the specified state, if it doesn't exist.

        Args:
            state (ndarray): The state to load.

        Returns:
            (str): The transformed state to index the value map.
        """

        state_t = self._transform_state(state)

        if state_t not in self._value_map:
            self._value_map[state_t] = np.asarray(
                [self._init_value] * self.mdp.n_action
            )

        return state_t

    def _transform_state(self, state: np.ndarray) -> str:
        """
        Transforms the specified state to a hashable format.

        Args:
            state (ndarray): The state to transform.

        Returns:
            (str): The transformed state.
        """

        assert isinstance(state, np.ndarray), (
            f"state must be of type {type(np.ndarray)}, not {type(state)}"
        )

        return ",".join(list(map(str, state.flatten())))
