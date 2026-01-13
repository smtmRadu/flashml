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
        path (str): Path to the jsonl file (including directories).
        add_timestamp (bool, optional): Whether to add a timestamp. Defaults to False.
        mode (str, optional): File mode, defaults to "a" (append).
        utf (str, optional): Encoding, defaults to "utf-8".
    """
    if not path.endswith((".jsonl", ".json")):
        path += ".jsonl"

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"[ℹ️ log_json] Directory created: {directory}")

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
) -> list[dict] | None:
    """
    Loads a jsonl file safely. 
    if as_df == False, it returns list[dict]
    """
    import os, json

    if not path.endswith((".jsonl", ".json")):
        path += ".jsonl"

    if not os.path.exists(path):
        from flashml import bell
        bell()
        print(
            f"⚠️  \033[93mThe file at path {path} couldn't be found, the returned object is None.\033[0m"
        )
        return None

    if os.stat(path).st_size == 0:
        return None

    records = []

    try:
        with open(path, "r", encoding=utf, errors="surrogatepass") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    # 🔥 Remove invalid UTF-16 surrogate chars
                    clean = (
                        line.encode("utf-16", "surrogatepass")
                        .decode("utf-16", "ignore")
                    )
                    records.append(json.loads(clean))
                except Exception as e:
                    print(
                        f"⚠️  Skipping line {lineno} due to decode/JSON error: {e}"
                    )

        if not records:
            return None

        if as_df is False or as_df == "list_of_dicts":
            return records

        elif as_df == "pd":
            import pandas as pd
            return pd.DataFrame(records)

        elif as_df == "pl":
            import polars as pl
            return pl.from_dicts(records)

        else:
            raise ValueError("Unhandled dataframe type.")

    except Exception as e:
        print(f"❌ \033[91mFailed to load {path}: {e}\033[0m")
        raise
