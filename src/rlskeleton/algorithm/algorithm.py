"""A module to manage the creation and usage of RL algorithms."""

import logging
from abc import ABC
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
from pydantic import BaseModel

from rlskeleton.function import Function
from rlskeleton.mdp import MDP


class AlgorithmParams(BaseModel):
    """Configuration parameters for an `Algorithm` object."""

    decay_delay: Optional[tuple[str, Any]] = None
    episodes: int


class Algorithm(ABC):
    """An abstract class representing an RL algorithm."""

    def __init__(self, mdp: MDP, function: Function, params: AlgorithmParams) -> None:
        """
        Instantiates a new `Algorithm` object.

        Args:
            mdp (MDP): The MDP this algorithm will interact with.
            function (Function): The policy function this algorithm will use.
            params (AlgorithmParams): The configuration parameters for this algorithm.
        """

        self.logger = logging.getLogger(type(self).__name__)

        self.function = function

        self.mdp = mdp

        self._params = params

    @property
    def config_params(self) -> AlgorithmParams:
        """Configuration parameters for this algorithm."""

        return self._params

    @property
    def start_episode_callback(self) -> Callable:
        return self._start_episode_callback

    @start_episode_callback.setter
    def start_episode_callback(self, callback: Callable) -> None:
        self._start_episode_callback = callback

    @property
    def start_step_callback(self) -> Callable:
        return self._start_step_callback

    @start_step_callback.setter
    def start_step_callback(self, callback: Callable) -> None:
        self._start_step_callback = callback

    @property
    def end_step_callback(self) -> Callable:
        return self._end_step_callback

    @end_step_callback.setter
    def end_step_callback(self, callback: Callable) -> None:
        self._end_step_callback = callback

    @property
    def end_episode_callback(self) -> Callable:
        return self._end_episode_callback

    @end_episode_callback.setter
    def end_episode_callback(self, callback: Callable) -> None:
        self._end_episode_callback = callback

    def learn(self) -> None:
        """Runs a learning session."""

        for episode in range(1, self._params.episodes + 1):
            self._start_episode(episode)

            try:
                self._run_episode(episode)
            finally:
                self._end_episode(episode)

    def _start_episode(self, episode: int) -> None:
        """
        Perform start of episode operations.

        Args:
            episode (int): The current episode.
        """

        if hasattr(self, "_start_episode_callback"):
            self._start_episode_callback(episode=episode)

        self._t1 = datetime.now(timezone.utc)

    def _run_episode(self, episode: int) -> None:
        """
        Run the specified episode.

        Args:
            episode (int): The current episode.
        """

        is_terminal = False

        state = self.mdp.start()

        while not is_terminal:
            state, is_terminal = self._run_step(episode, state)

    def _end_episode(self, episode: int) -> None:
        """
        Perform end of episode operations.

        Args:
            episode (int): The current episode.
        """

        elapsed_time = datetime.now(timezone.utc) - self._t1

        self._decay(episode)

        if hasattr(self, "_end_episode_callback"):
            self._end_episode_callback(episode=episode, elapsed_time=elapsed_time)

        del self._t1

    def _run_step(self, episode: int, state: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Run a step for the specified episode.

        Args:
            episode (int): The current episode.
            state (ndarray): The current state.

        Returns:
            The next state and whether or not it's a terminal state.
        """

        if hasattr(self, "_start_step_callback"):
            self._start_step_callback(episode=episode, state=state)

        action = self.function.choose_action(state)

        reward, next_state, is_terminal, info = self.mdp.step(action)

        self._end_step(episode, state, action, next_state, reward, info)

        return next_state, is_terminal

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
        Performs the required post-step actions for this algorithm.

        Args:
            episode (int): The current episode.
            state (ndarray): The current state.
            action (int): The action taken at the current state.
            next_state (ndarray): The resulting state from the action taken.
            reward (float): The reward resulting from the action taken.
            info (dict[str, Any]): Additional information.
        """

        if hasattr(self, "_end_step_callback"):
            self._end_step_callback(
                episode=episode,
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                info=info,
            )

    def _decay(self, episode: int) -> None:
        """
        Executes the decay for this algorithm's value function.

        Args:
            episode (int): The current episode.
        """

        run_decay = False

        if self._params.decay_delay:
            if self._params.decay_delay[0] == "episode":
                if episode > int(self._params.decay_delay[1]):
                    run_decay = True
            else:
                raise RuntimeError(
                    f"Invalid decay delay type: {self._params.decay_delay[0]}"
                )
        else:
            run_decay = True

        if run_decay:
            self.function.decay(episode)

    @staticmethod
    def parse_config_params(obj: Any) -> AlgorithmParams:
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

        return AlgorithmParams(
            decay_delay=decay_delay,
            episodes=obj["episodes"],
        )
