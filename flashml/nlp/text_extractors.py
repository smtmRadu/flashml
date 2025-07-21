def extract_text_within_tags(df, column_name, tag_name, join_matches_by='\n'):
    """Extracts the text within specified HTML/XML tags from a column in Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame.
        column_name: str - name of the column to process.
        tag_name: str - name of the tag to extract text from.

    Returns:
        A DataFrame with the text within specified tags extracted and replaced in the given column. If nothing is found, sets the field to None.
    """
    import re

    pattern = fr'<{tag_name}>(.*?)</{tag_name}>'

    if df.__class__.__module__.startswith("pandas"):
        def extract_or_none(x):
            matches = re.findall(pattern, x, re.DOTALL)
            return join_matches_by.join(matches) if matches else None

        df[column_name] = df[column_name].astype(str).apply(extract_or_none)
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        def extract_or_none(x):
            matches = re.findall(pattern, x, re.DOTALL)
            return join_matches_by.join(matches) if matches else None

        return df.with_columns(
            pl.col(column_name).map_elements(extract_or_none, return_dtype=pl.Utf8).alias(column_name)
        )

    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")
