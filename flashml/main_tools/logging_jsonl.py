import json
import os
from datetime import datetime
from typing import Literal


def log_record(
    record: dict | str,
    path="flashml_logger.jsonl",
    add_timestamp=False,
    mode: str = "a",
    utf="utf-8",
):
    """Records a dictionary as a json object in a jsonl file.

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


def load_records(
    path="flashml_logger.jsonl",
    as_df: Literal["pd", "pl"] = "list_of_dicts",
    utf="utf-8",
) -> list[dict]:
    """Loads the jsonl file and returns a polars/pandas dataframe.

    Returns:
        list[dict] | polars/pandas df | **None** if file is empty.
    """
    # check file is empty
    if not os.path.exists(path):
        print(
            f"\033[93mThe file at path {path} couldn't be found, the returned object is None.\033[0m"
        )

        return None
        # raise "File does not exist."

    if os.stat(path).st_size == 0:
        return None

    if as_df == "list_of_dicts":
        import pandas

        r = pandas.read_json(path, lines=True, encoding=utf)
        return r.to_dict(orient="records")
    elif as_df == "pd":
        import pandas

        return pandas.read_json(path, lines=True, encoding=utf)
    elif as_df == "pl":
        import polars

        return polars.read_ndjson(path)
    else:
        raise "Unhandled dataframe type."
