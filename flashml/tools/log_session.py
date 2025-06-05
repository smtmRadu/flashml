from datetime import datetime
import os
from typing import Any, Callable, Dict


class _TrainingLogger:
    _instance = None
    _output_file = "session_results.md"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.metric_columns = set()  # Tracks all metric names
        self.hyperparam_columns = set()  # Tracks all hyperparameter names

    def _read_existing_table(self):
        import polars as pl

        """Read the existing markdown table into a Polars DataFrame."""
        if not os.path.exists(self._output_file):
            return None

        with open(self._output_file, "r") as f:
            lines = f.readlines()

        table_lines = [line.strip() for line in lines if line.strip().startswith("|")]
        if len(table_lines) < 2:
            return None

        headers = [h.strip("`").strip() for h in table_lines[0].split("|")[1:-1]]
        rows = []
        for line in table_lines[2:]:
            row = [r.strip() for r in line.split("|")[1:-1]]
            rows.append(row)

        if not rows:
            return None

        return pl.DataFrame(
            {col: [row[i] for row in rows] for i, col in enumerate(headers)}
        )

    def _log_session(
        self,
        hyperparameters: dict[str, Any] | None,
        train_func: Callable[[], Dict[str, Any]] | None,
        sort_by: str | None,
    ):
        import polars as pl

        """
        Log a training session or sort the existing table.

        Args:
            hyperparameters (dict[str, Any] | None): Dictionary of hyperparameters.
            train_func (Callable[[], Dict[str, Any]] | None): Function returning a dict of metrics.
            sort_by (str | None): Column name to sort by.
        """
        existing_df = self._read_existing_table()

        # If no table exists and no new data, exit
        if existing_df is None and hyperparameters is None and train_func is None:
            print("\033[93mNo existing table to sort.\033[0m")
            return

        # Create an empty DataFrame if none exists
        if existing_df is None:
            existing_df = pl.DataFrame()

        # Sorting-only mode
        if hyperparameters is None and train_func is None:
            if sort_by and sort_by in existing_df.columns:
                try:
                    existing_df = existing_df.with_columns(
                        pl.col(sort_by)
                        .cast(str)
                        .replace("N/A", None)
                        .cast(pl.Float64, strict=False)
                        .alias(sort_by)
                    ).sort(sort_by, descending=True, nulls_last=True)
                except Exception as e:
                    print(f"\033[93mFailed to sort by '{sort_by}'. Error: {e}\033[0m")
            self._write_table(existing_df, sort_by, is_new_session=False)
            return

        # Logging a new session
        start_time = datetime.now()
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

        score_result = train_func()  # Dictionary of metrics
        end_time = datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = (end_time - start_time).total_seconds()

        # Build session dictionary
        session = {
            "Start Time": start_time_str,
            "End Time": end_time_str,
            "Duration": f"{elapsed_time:.2f}",
        }

        # Add metrics directly (no '@' prefix)
        session.update(score_result)
        self.metric_columns.update(score_result.keys())

        # Add hyperparameters
        if hyperparameters:
            session.update(hyperparameters)
            self.hyperparam_columns.update(hyperparameters.keys())

        # Combine with existing sessions
        all_sessions = (
            (
                [s for s in existing_df.drop("Index").to_dicts()]
                if "Index" in existing_df.columns
                else existing_df.to_dicts()
            )
            + [session]
            if len(existing_df) > 0
            else [session]
        )

        # Fill missing columns with "N/A"
        all_columns = {col for s in all_sessions for col in s.keys()}
        for s in all_sessions:
            for col in all_columns:
                s.setdefault(col, "N/A")

        df = pl.DataFrame(all_sessions)

        # Define column order
        fixed_cols = ["Start Time", "End Time", "Duration"]
        metric_cols = sorted(
            [
                col
                for col in df.columns
                if col in self.metric_columns and col not in fixed_cols
            ]
        )
        hyper_cols = sorted(
            [
                col
                for col in df.columns
                if col in self.hyperparam_columns
                and col not in self.metric_columns
                and col not in fixed_cols
            ]
        )
        other_cols = sorted(
            [
                col
                for col in df.columns
                if col not in fixed_cols
                and col not in metric_cols
                and col not in hyper_cols
            ]
        )
        column_order = fixed_cols + metric_cols + hyper_cols + other_cols

        df = df.select(column_order)

        # Sort if requested
        if sort_by and sort_by in df.columns:
            try:
                df = df.with_columns(
                    pl.col(sort_by)
                    .cast(str)
                    .replace("N/A", None)
                    .cast(pl.Float64, strict=False)
                    .alias(sort_by)
                ).sort(sort_by, descending=True, nulls_last=True)
            except Exception as e:
                print(f"\033[93mFailed to sort by '{sort_by}'. Error: {e}\033[0m")

        self._write_table(df, sort_by, is_new_session=True)

    def _write_table(self, df, sort_by: str | None, is_new_session: bool = True):
        import polars as pl

        """Write the DataFrame to a markdown file."""
        # Fill null values with "N/A" for all columns
        df = df.with_columns([pl.col(col).fill_null("N/A") for col in df.columns])

        # Drop existing "Index" column if it exists to avoid DuplicateError
        if "Index" in df.columns:
            df = df.drop("Index")

        # Add a new "Index" column and reorder columns to put "Index" first
        df = df.with_row_index("Index").select(
            ["Index"] + [col for col in df.columns if col != "Index"]
        )

        # Format headers, adding backticks to the sorted column
        headers = [f"`{h}`" if sort_by == h else h for h in df.columns]

        # Construct markdown table
        markdown_content = (
            "|"
            + "|".join(headers)
            + "|\n"
            + "|"
            + "|".join(["---"] * len(headers))
            + "|\n"
        )
        markdown_content += (
            "\n".join(
                "|" + "|".join(str(val) for val in row) + "|" for row in df.rows()
            )
            + "\n"
        )

        # Write to file
        with open(self._output_file, "w") as f:
            f.write(markdown_content)

        # Print confirmation message
        print(
            f"\033[92m{'Session logged' if is_new_session else 'Table sorted'} to {self._output_file}\033[0m"
        )


def log_session(
    hyperparameters: dict[str, Any] | None = None,
    train_func: Callable[[], Dict[str, Any]] | None = None,
    sort_by: str | None = None,
) -> None:
    """
    Log a training session to a markdown file. When sorting by a column, the name of that columns is highlighted.

    Args:
        hyperparameters (dict[str, Any] | None): Hyperparameters for the session.
        train_func (Callable[[], Dict[str, Any]] | None): Function returning metrics.
        sort_by (str | None): Column to sort by.
    """
    logger = _TrainingLogger()
    logger._log_session(hyperparameters, train_func, sort_by)
