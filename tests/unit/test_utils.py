"""Tests for utilities module."""

from datetime import datetime, timedelta, timezone
from enum import Enum

import rlskeleton.utils as utils


def test_json_serializer_datetime() -> None:
    value = datetime.now(timezone.utc)

    expected = value.isoformat()

    actual = utils.json_serializer(value)

    assert actual == expected


def test_json_serializer_timedelta() -> None:
    expected = 12.6

    value = timedelta(seconds=expected)

    actual = utils.json_serializer(value)

    assert actual == expected


def test_json_serializer_enum() -> None:
    expected = "alpha"

    class Thing(Enum):
        ALPHA = expected

    value = Thing.ALPHA

    actual = utils.json_serializer(value)

    assert actual == expected
