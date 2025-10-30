"""An module to manage IO operations."""

import os
from io import BytesIO

from .registry import Registry


class LocalRegistry(Registry):
    """A class representing a registry on local disk."""

    def __init__(self, root: str) -> None:
        """
        Instantiates a new `LocalRegistry` object.

        Args:
            root (str): The root path where artifacts will be read from and written to.
        """

        super().__init__()

        self.root = root

    def read_bytes(self, path: str | list[str]) -> BytesIO:
        """
        Reads bytes from an object in the registry.

        Args:
            path (str | list[str]): The path of or path parts to the object in the registry.

        Returns:
            (BytesIO): The bytes from the registry at the specified path.
        """

        file_path = self._validate_path(path)

        with open(file_path, mode="rb") as fd:
            buffer = BytesIO(fd.read())

        return buffer

    def write_bytes(self, path: str | list[str], buffer: BytesIO) -> None:
        """
        Writes bytes to an object in the registry.

        Args:
            path (str | list[str]): The path of or path parts to the object in the registry.
            buffer (BytesIO): The bytes of the object to write to the registry.
        """

        file_path = self._validate_path(path)

        with open(file_path, mode="wb") as fd:
            fd.write(buffer.getvalue())

    def append_bytes(self, path: str | list[str], buffer: BytesIO) -> None:
        """
        Appends bytes to an object in the registry.

        Args:
            path (str | list[str]): The path of or path parts to the object in the registry.
            buffer (BytesIO): The bytes of the object to append to in the registry.
        """

        file_path = self._validate_path(path)

        with open(file_path, mode="ab") as fd:
            fd.write(buffer.getvalue())

    def _validate_path(self, path: str | list[str]) -> str:
        """
        Creates the specified path if it doesn't exist and returns the fully joined path.

        Args:
            path (str | list[str]): The path of or path parts to the object in the registry.

        Returns:
            (str): The joined path.
        """

        prefix = "./" if self.root[0] != "." and self.root[0] != "/" else ""

        if isinstance(path, list):
            dir = os.path.join(*[prefix, self.root] + path[:-1])

            valid_path = os.path.join(*[prefix, self.root] + path)
        else:
            dir, filename = os.path.split(path)

            valid_path = os.path.join(prefix, self.root, dir, filename)

        os.makedirs(dir, exist_ok=True)

        return valid_path
