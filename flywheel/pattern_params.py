"""Pattern parameter coercion and placeholder substitution."""

from __future__ import annotations

import re
from typing import Any, Literal

ParamType = Literal["string", "int", "float", "bool"]

_PARAM_PLACEHOLDER_RE = re.compile(
    r"\$\{params\.([A-Za-z0-9][A-Za-z0-9_-]*)\}"
)


class PatternParamError(ValueError):
    """Raised for invalid pattern parameter declarations or values."""


def referenced_params(value: str) -> set[str]:
    """Return parameter names referenced by one string."""
    return set(_PARAM_PLACEHOLDER_RE.findall(value))


def format_param_value(value: object) -> str:
    """Render a parameter value for env/argv substitution."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def substitute_params(value: str, params: dict[str, object]) -> str:
    """Substitute ${params.name} placeholders in one string."""
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            raise PatternParamError(f"unknown pattern param {name!r}")
        return format_param_value(params[name])

    return _PARAM_PLACEHOLDER_RE.sub(replace, value)


def coerce_param_value(
    *,
    pattern_name: str,
    name: str,
    value: Any,
    param_type: ParamType,
    source: str,
) -> str | int | float | bool:
    """Coerce a declared/default/override value to a parameter type."""
    try:
        if param_type == "string":
            if not isinstance(value, str):
                raise TypeError
            return value
        if param_type == "int":
            if isinstance(value, bool):
                raise TypeError
            return int(value)
        if param_type == "float":
            if isinstance(value, bool):
                raise TypeError
            return float(value)
        if param_type == "bool":
            return _coerce_bool(value)
    except (TypeError, ValueError) as exc:
        raise PatternParamError(
            f"Pattern {pattern_name!r}: param {name!r} {source} "
            f"{value!r} is not a valid {param_type}"
        ) from exc
    raise AssertionError(param_type)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    raise TypeError
