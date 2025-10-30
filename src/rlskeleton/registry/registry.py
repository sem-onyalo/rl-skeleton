"""An module to manage IO operations to the local disk."""

import logging
from abc import ABC
from io import BytesIO


class Registry(ABC):
    """An abstract class representing a registry."""

    def __init__(self) -> None:
        """Instantiates a new `Registry` object."""

        self.logger = logging.getLogger(type(self).__name__)

    def read_bytes(self, path: str | list[str]) -> BytesIO:
        """
        Reads bytes from an object in the registry.

        Args:
            path (str | list[str]): The path of or path parts to the object in the registry.

        Returns:
            The bytes from the registry at the specified path.
        """

        raise RuntimeError("Subclass must implement function.")

    def write_bytes(self, path: str | list[str], buffer: BytesIO) -> None:
        """
        Writes bytes to an object in the registry.

        Args:
            path (str | list[str]): The path of or path parts to the object in the registry.
            buffer (BytesIO): The bytes of the object to write to in the registry.
        """

        raise RuntimeError("Subclass must implement function.")

    def append_bytes(self, path: str | list[str], buffer: BytesIO) -> None:
        """
        Appends bytes to an object in the registry.

        Args:
            path (str | list[str]): The path of or path parts to the object in the registry.
            buffer (BytesIO): The bytes of the object to append to in the registry.
        """

        raise RuntimeError("Subclass must implement function.")
