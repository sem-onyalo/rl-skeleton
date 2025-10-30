"""A general utilities module."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any


def json_serializer(obj: Any) -> Any:
    """JSON Serializer for `Function` dependent objects."""

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, timedelta):
        return obj.total_seconds()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
