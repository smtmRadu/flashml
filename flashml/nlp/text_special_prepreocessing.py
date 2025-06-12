def expand_contractions(df, column_name):
    import contractions

    """
    Expands contractions in a text column in a Pandas or Polars DataFrame using the contractions library.
    """

    def _expand_func(text):
        if text is None:
            return None
        return contractions.fix(str(text))

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].apply(_expand_func)
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .map_elements(_expand_func, return_dtype=pl.Utf8)
            .alias(column_name)
        )

    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def lowercase(df, column_name):
    """
    Lowercases a column in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - the name of the column to lowercase

    Returns:
        A new DataFrame with the specified column lowercased
    """
    if df.__class__.__module__.startswith("pandas"):
        # Ensure the column is treated as string
        df[column_name] = df[column_name].astype(str).str.lower()
        return df
    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            df[column_name].cast(pl.Utf8).str.to_lowercase().alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def lemmatize(df, column_name, spacy_model="en_core_web_sm"):
    """
    Applies lemmatization to a text column in a DataFrame using spaCy.

    Args:
        df: A pandas.DataFrame or polars.DataFrame.
        column_name: str - The name of the column to process.
        model: A loaded spaCy language model.

    Returns:
        A DataFrame with lemmatized text in the specified column.
    """
    import spacy

    try:
        nlp_lemma = spacy.load(spacy_model)
    except:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "spacy", "download", f"{spacy_model}"], check=True
        )
        nlp_lemma = spacy.load(spacy_model)

    if nlp_lemma is None:
        raise ValueError("A loaded spaCy model must be provided for lemmatization.")

    def _lemmatize_text(text):
        if text is None:
            return None
        doc = nlp_lemma(str(text))
        return " ".join([token.lemma_ for token in doc])

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].apply(_lemmatize_text)
        return df
    if df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .map_elements(_lemmatize_text, return_dtype=pl.Utf8)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")
