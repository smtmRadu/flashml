def remove_stopwords(df, column_name, model_spacy="en_core_web_sm"):
    """
    Removes stopwords from a text column in a Pandas or Polars DataFrame using spaCy.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
    Returns:
        A DataFrame with the specified column having stopwords removed
    """
    import spacy

    try:
        nlp = spacy.load(model_spacy, disable=["ner", "parser"])  # lightweight
    except:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_spacy], check=True
        )
        nlp = spacy.load(model_spacy, disable=["ner", "parser"])

    def _remove_stopwords_func(text):
        if text is None:
            return None
        try:
            doc = nlp(str(text))
            filtered_words = [token.text for token in doc if not token.is_stop]
            return " ".join(filtered_words)
        except Exception:
            return str(text)

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].apply(_remove_stopwords_func)
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .map_elements(_remove_stopwords_func, return_dtype=pl.Utf8)
            .alias(column_name)
        )

    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def remove_double_spacing(df, column_name):
    """
    Removes multiple consecutive spaces from a text column in a Pandas or Polars DataFrame, replacing them with a single space.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process

    Returns:
        A DataFrame with multiple spaces reduced to single spaces in the specified column
    """
    # Regex to match two or more whitespace characters
    space_pattern = r"\s+"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name].astype(str).str.replace(space_pattern, " ", regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(space_pattern, " ")
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")
