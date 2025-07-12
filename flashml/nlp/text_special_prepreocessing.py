
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


def lemmatize(
    df,
    column_name: str,
    multiprocess: int = 1,
    batch_size: int = 32,
    spacy_model: str = "en_core_web_sm",
):
    """
    Applies parallel lemmatization to a text column in a DataFrame using spaCy.
    This works only in .py files, when called within if __name__ == "__main__":

    Args:
        df: A pandas.DataFrame or polars.DataFrame.
        column_name: The name of the column to process.
        spacy_model: The name of the spaCy model to use.
        batch_size: The number of texts to buffer during processing.

    Returns:
        A new DataFrame with the lemmatized text in the specified column.
    """

    import spacy
    import sys
    import subprocess

    assert multiprocess > 0

    try:
        nlp = spacy.load(spacy_model, disable=["parser", "ner"])
    except OSError:
        print(f"Spacy model '{spacy_model}' not found. Downloading...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", spacy_model], check=True
        )
        nlp = spacy.load(spacy_model, disable=["parser", "ner"])

    if df.__class__.__module__.startswith("pandas"):
        texts = df[column_name].tolist()
    if df.__class__.__module__.startswith("polars"):
        texts = df[column_name].to_list()

    if multiprocess == 1:

        def _lemmatize_text(text):
            if text is None:
                return None
            doc = nlp(str(text))
            return " ".join([token.lemma_ for token in doc])

        if df.__class__.__module__.startswith("pandas"):
            df[column_name] = df[column_name].apply(_lemmatize_text)
            return df
        if df.__class__.__module__.startswith("polars"):
            import polars as pl

            return df.with_columns(
                pl.col(column_name)
                .map_elements(
                    _lemmatize_text, return_dtype=pl.Utf8, strategy="thread_local"
                )
                .alias(column_name)
            )
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")
    else:
        docs = nlp.pipe(
            [str(text) if text is not None else "" for text in texts],
            batch_size=batch_size,
            n_process=multiprocess,
            disable=["parser", "ner"],
        )
        lemmatized_texts = [" ".join([token.lemma_ for token in doc]) for doc in docs]

        # 3. Return a new DataFrame of the original type with the updated column.
        if df.__class__.__module__.startswith("pandas"):
            return df.assign(**{column_name: lemmatized_texts})
        if df.__class__.__module__.startswith("polars"):
            import polars as pl

            return df.with_columns(pl.Series(name=column_name, values=lemmatized_texts))
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")
