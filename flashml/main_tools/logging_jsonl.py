import json
import os
from datetime import datetime
from typing import Literal


def log_json(
    record: dict | str,
    path="flashml_logger.jsonl",
    add_timestamp=False,
    mode: str = "a",
    utf="utf-8",
):
    """Logs a dictionary as a json object in a jsonl file.

    Args:
        record (dict | str): A message or a dictionary
        path (flashml_logger.jsonl): _description_
        mode (str, optional): _description_. Defaults to "a".
        utf (str, optional): _description_. Defaults to "utf-8".
    """
    if isinstance(record, str):
        record = {"message": record}
    if add_timestamp:
        new_dict = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **record}
    else:
        new_dict = record

    with open(path, mode, encoding=utf) as f:
        f.write(json.dumps(new_dict) + "\n")


def load_jsonl(
    path="flashml_logger.jsonl",
    as_df: Literal["pd", "pl"] = False,
    utf="utf-8",
) -> list[dict]:
    """Loads the jsonl file and returns a polars/pandas dataframe.

    Args:
        path (str, optional): Path to the jsonl file. Defaults to "flashml_logger.jsonl".
        as_df (Literal["pd", "pl"], optional): Return a dataframe. Defaults to False. If false, returns a list of dictionaries.
        utf (str, optional): File encoding. Defaults to "utf-8".
    Returns:
        list[dict] | polars/pandas df | None if file is empty.
    """
    import os, json

    # check file exists
    if not os.path.exists(path):
        from flashml import bell
        bell()
        print(
            f"\033[93mWARNING: The file at path {path} couldn't be found, the returned object is None.\033[0m"
        )
        return None

    if os.stat(path).st_size == 0:
        return None

    try:
        if as_df is False or as_df == "list_of_dicts":
            import pandas as pd
            r = pd.read_json(path, lines=True, encoding=utf)
            return r.to_dict(orient="records")
        elif as_df == "pd":
            import pandas as pd
            return pd.read_json(path, lines=True, encoding=utf)
        elif as_df == "pl":
            import polars as pl
            return pl.read_ndjson(path)
        else:
            raise ValueError("Unhandled dataframe type.")

    except ValueError as e:
        # Fallback: scan line by line to find bad JSON
        print(f"\033[91mError while reading {path}: {e}\033[0m")
        print("Scanning file line by line to locate issue...\n")
        bad_lines = []
        with open(path, "r", encoding=utf) as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as je:
                    bad_lines.append((lineno, je.msg, line.strip()[:200]))
        if bad_lines:
            for lineno, msg, sample in bad_lines:
                print(f"Line {lineno}: {msg}\n  {sample}\n")
        else:
            print("No obvious bad lines found (may be an encoding or quoting issue).")
        raise