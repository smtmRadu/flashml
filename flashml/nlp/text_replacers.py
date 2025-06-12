def replace_numbers(df, column_name, replacement="NUM", including_floats=True):
    """
    Replaces numbers in a text column with a placeholder in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace numbers with (default: 'NUM')

    Returns:
        A DataFrame with numbers replaced in the specified column
    """
    # Regex to find one or more digits
    number_pattern = r"\d+" if not including_floats else r"-?\b\d+(?:[\.,]\d+)?\b"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(number_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(number_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def replace_urls(df, column_name, replacement="URL"):
    """
    Replaces URLs in a text column with a placeholder in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace URLs with (default: 'URL')

    Returns:
        A DataFrame with URLs replaced in the specified column
    """
    # Regex to find URLs
    url_pattern = r"http\S+|www\S+|https\S+"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(url_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(url_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def replace_emails(df, column_name, replacement="EMAIL"):
    """
    Replaces email addresses in a text column with a placeholder in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace emails with (default: 'EMAIL')

    Returns:
        A DataFrame with email addresses replaced in the specified column
    """
    # Regex to match email addresses
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(email_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(email_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def replace_punctuation(df, column_name, replacement="PUNC"):
    """
    Replaces punctuation in a text column with a placeholder in a Pandas or Polars DataFrame.
    Replaces sequences of one or more punctuation characters with a single replacement token.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace punctuation with (default: 'PUNC')

    Returns:
        A DataFrame with punctuation replaced in the specified column
    """
    import re
    import string

    # Create a regex pattern that matches one or more punctuation characters
    # Use re.escape to handle special characters in string.punctuation correctly
    punct_pattern = f"[{re.escape(string.punctuation)}]+"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(punct_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(punct_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def replace_pattern(df, column_name, regex_pattern, replacement="MATCH"):
    """
    GENERAL - Replaces text matching a regex pattern in a text column of a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        pattern: str - regex pattern to match
        replacement: str - the string to replace matches with (default: 'MATCH')

    Returns:
        A DataFrame with the matched patterns replaced in the specified column
    """
    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(regex_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(regex_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def replace_hashtags(df, column_name, replacement="HASH"):
    """
    Replaces hashtags (e.g. #ab4_23a) in a text column with a placeholder in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace hashtags with (default: 'HASH')

    Returns:
        A DataFrame with hashtags replaced in the specified column
    """
    # Regex to match hashtags: # followed by letters, numbers, or underscores
    hashtag_pattern = r"#[\w]+"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(hashtag_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(hashtag_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def replace_emojis(df, column_name, replacement="EMOJI"):
    """
    Replaces emojis in a text column with a placeholder in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace emojis with (default: 'EMOJI')

    Returns:
        A DataFrame with emojis replaced in the specified column
    """
    # Regex pattern to match common emojis (covers most Unicode emoji ranges)
    emoji_pattern = (
        r"[\U0001F600-\U0001F64F"  # Emoticons
        r"\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
        r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        r"\U0001F700-\U0001F77F"  # Alchemical Symbols
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        r"\U00002600-\U000026FF"  # Miscellaneous Symbols (e.g., ☀️)
        r"\U00002700-\U000027BF]"  # Dingbats (e.g., ✂️)
    )

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(emoji_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(emoji_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError(
            "Unsupported 不过另外一邊是什麼意思 Unsupported DataFrame type. Must be pandas or polars."
        )


def replace_smileys(df, column_name, replacement="SMILEY"):
    """
    Replaces text-based smiley faces (e.g., :), :-), =)))))) in a text column with a placeholder in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace smileys with (default: 'SMILEY')

    Returns:
        A DataFrame with smiley faces replaced in the specified column
    """
    # Regex pattern for smiley faces like :), :-), =))))
    smiley_pattern = r"[:=]-?\)+"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(smiley_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(smiley_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def replace_currency(df, column_name, replacement="CURRENCY"):
    """
    Replaces currencies ($100, €50, ¥1000) in a text column with a placeholder in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace emails with (default: 'EMAIL')

    Returns:
        A DataFrame with email addresses replaced in the specified column
    """
    currency_pattern = r"[$€£¥₹]\s?-?\d+(?:[\.,]\d+)?|-?\d+(?:[\.,]\d+)?\s?[$€£¥₹]"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(currency_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(currency_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")


def replace_measurements(df, column_name, replacement="SIZE"):
    """
    Replaces measurements (8oz, 13kg etc.) in a text column with a placeholder in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace smileys with (default: 'SMILEY')

    Returns:
        A DataFrame with smiley faces replaced in the specified column
    """
    # Regex pattern for smiley faces like :), :-), =))))
    m_pattern = r"\b\d+(\.\d+)?\s?(oz|kg|g|lb|lbs|m|cm|mm|in|ft|yd|L|ml|gallon|pound)\b"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = (
            df[column_name].astype(str).str.replace(m_pattern, replacement, regex=True)
        )
        return df

    elif df.__class__.__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(
            pl.col(column_name)
            .cast(pl.Utf8)
            .str.replace_all(m_pattern, replacement)
            .alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")
