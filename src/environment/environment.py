import logging
from typing import Dict
from typing import Tuple

import numpy as np

class Environment:
    def __init__(self, name:str) -> None:
        self.name = name
        self.logger = logging.getLogger(name)

    def reset(self) -> np.ndarray:
        """
        Resets this `Environment` for a new session.

        :return The current state.
        """
        raise NotImplementedError()
    
    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        """
        Executes a step using the specified action in this `Environment`.

        :param action: The action to execute.
        :return The next state as a result of the specified action, 
            The reward as a result of the specified action,
            Whether or not the next state is a terminal state,
            Additional info as a result of the specified action
        """
        raise NotImplementedError()
