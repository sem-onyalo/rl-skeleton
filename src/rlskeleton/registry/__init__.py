"""An module to manage IO operations to the local disk."""

from .registry import Registry
from .local_registry import LocalRegistry

__all__ = ["Registry", "LocalRegistry"]
