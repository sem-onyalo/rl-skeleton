"""A module to manage the functionality of an RL agent."""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO

from .algorithm import Algorithm
from .registry import Registry
from .utils import json_serializer


class Agent:
    """A class representing an RL agent."""

    def __init__(self, algorithm: Algorithm, registry: Registry) -> None:
        """
        Instantiates a new `Agent` object.

        Args:
            algorithm (Algorithm): The algorithm this agent will use.
            registry (Registry): The registry this agent will use.
        """

        self.logger = logging.getLogger(type(self).__name__)

        self.algorithm = algorithm

        self.registry = registry

        self._run_params_file_name = "run-args.json"

        self._model_file_name = f"model{self.algorithm.function.model_file_ext}"

    def run(self) -> None:
        """Runs the agent."""

        self._start_run()

        try:
            self.algorithm.learn()
        finally:
            self._end_run()

        # TODO: act functionality
        # self.algorithm.act()

    def _start_run(self) -> None:
        """Performs start-of-learning operations."""

        self._run_id = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4()}"
        )

        self._save_params_run_start()

        self._t0 = datetime.now(timezone.utc)

    def _end_run(self) -> None:
        """Performs end-of-learning operations."""

        elapsed_time = datetime.now(timezone.utc) - self._t0

        self._save_model()

        self._save_params_run_end(elapsed_time=elapsed_time)

        self._log_params_run_end(elapsed_time)

        del self._t0
        del self._run_id

    def _log_params_run_end(self, elapsed_time: timedelta) -> None:
        """
        Log parameters before the end of the run.

        Args:
            elapsed_time (timedelta): The elapsed time of the run.
        """

        self.logger.info("-" * 50)
        self.logger.info(f"run ID: {self._run_id}")
        self.logger.info(f"run elapsed time: {elapsed_time}")

    def _save_model(self) -> None:
        """Saves the model to the registry."""

        path = [self._run_id, self._model_file_name]

        buffer = self.algorithm.function.get_model()

        self.registry.write_bytes(path, buffer)

    def _save_params_run_start(self) -> None:
        """Saves the start-of-run parameters to the registry."""

        agent_params = {
            "algorithm": type(self.algorithm).__name__,
            "mdp": type(self.algorithm.function.mdp).__name__,
            "function": type(self.algorithm.function).__name__,
            "run_id": self._run_id,
            "start_timestamp": datetime.now(timezone.utc),
        }

        params = (
            agent_params
            | self.algorithm.config_params.model_dump()
            | self.algorithm.function.config_params.model_dump()
        )

        path = [self._run_id, self._run_params_file_name]

        buffer = BytesIO()

        buffer.write(
            json.dumps(params, indent=4, default=json_serializer).encode("utf-8")
        )

        self.registry.write_bytes(path, buffer)

    def _save_params_run_end(self, elapsed_time: timedelta) -> None:
        """Saves the end-of-run parameters to the registry."""

        path = [self._run_id, self._run_params_file_name]

        buffer = self.registry.read_bytes(path)

        params = json.load(buffer)

        run_params = {
            "elapsed_time": elapsed_time,
            "end_timestamp": datetime.now(timezone.utc),
        }

        params = params | run_params

        buffer.seek(0)

        buffer.write(
            json.dumps(params, indent=4, default=json_serializer).encode("utf-8")
        )

        self.registry.write_bytes(path, buffer)
