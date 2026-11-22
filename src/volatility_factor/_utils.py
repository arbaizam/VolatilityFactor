"""Small internal helpers shared across modules."""

from __future__ import annotations

import datetime as _dt
import functools
import re
from typing import Any

import pandas as pd

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_date(value: Any, name: str, *, optional: bool = False) -> str | None:
    """Return ``value`` as a ``YYYY-MM-DD`` string, or raise ``ValueError``.

    Strings must already be ``YYYY-MM-DD``. ``datetime``/``date``/``Timestamp`` values are
    formatted to that string.
    """
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} is required (YYYY-MM-DD)")
    if isinstance(value, (pd.Timestamp, _dt.date)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, str) and _DATE_RE.match(value):
        return value
    suffix = "None or " if optional else ""
    raise ValueError(f"{name} must be {suffix}a date string formatted as YYYY-MM-DD; got {value!r}")


def clear_cached_properties(obj: object) -> None:
    """Drop every ``functools.cached_property`` value cached on ``obj``."""
    for klass in type(obj).__mro__:
        for name, attr in vars(klass).items():
            if isinstance(attr, functools.cached_property):
                obj.__dict__.pop(name, None)
