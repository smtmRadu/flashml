
import ast
import sys
from typing import Any


def _convert_arg_value(value: str) -> Any:
    """
    Convert CLI strings to Python values when possible.

    Examples:
        "2" -> 2
        "2.5" -> 2.5
        "true" -> True
        "none" -> None
        "[1, 2]" -> [1, 2]
    """
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        if "." not in value and "e" not in lowered:
            return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _is_number_token(token: str) -> bool:
    """Return True when a token like -1 or -0.5 is a value, not an option."""
    try:
        float(token)
        return True
    except ValueError:
        return False


def _is_option_token(token: str) -> bool:
    """Return True for CLI option tokens such as --lr or -n."""
    if not token or token == "-":
        return False
    if not token.startswith("-"):
        return False
    if token == "--":
        return True
    return not _is_number_token(token)


def load_args() -> dict[str, Any]:
    """
    Parse command line arguments into a dictionary.

    Supported forms:
        --key value
        --key=value
        --flag
        --items 1 2 3
        -k value

    Notes:
        - Hyphens in option names are converted to underscores.
        - Flags without values become True.
        - Options prefixed with --no- become False.
        - Positional arguments are stored under "_args" when present.
    """
    tokens = list(sys.argv[1:])
    parsed: dict[str, Any] = {}
    positional_args: list[Any] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if token == "--":
            positional_args.extend(_convert_arg_value(item) for item in tokens[index + 1 :])
            break

        if not _is_option_token(token):
            positional_args.append(_convert_arg_value(token))
            index += 1
            continue

        if "=" in token:
            key, raw_value = token.lstrip("-").split("=", 1)
            parsed[key.replace("-", "_")] = _convert_arg_value(raw_value)
            index += 1
            continue

        key = token.lstrip("-").replace("-", "_")
        index += 1
        values: list[Any] = []

        while index < len(tokens):
            next_token = tokens[index]
            if next_token == "--" or _is_option_token(next_token):
                break
            values.append(_convert_arg_value(next_token))
            index += 1

        if not values:
            if key.startswith("no_") and len(key) > 3:
                parsed[key[3:]] = False
            else:
                parsed[key] = True
            continue

        parsed[key] = values[0] if len(values) == 1 else values

    if positional_args:
        parsed["_args"] = positional_args

    return parsed
