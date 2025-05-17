from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import re
import polars as pl





def expand_contractions(df, column_name):
    """
    Expands contractions in a text column in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process

    Returns:
        A DataFrame with contractions expanded in the specified column
    """
    # Simple contraction dictionary (expandable)
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",  # General rule for "not" contractions
        "i'm": "i am",
        "you're": "you are",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "'ll": " will",  # General rule for "will" contractions
        "'ve": " have",  # General rule for "have" contractions
        "'d": " would",  # General rule for "would" contractions
    }

    def _expand_func(text):
        if text is None:
            return None
        text = str(text).lower()  # Case-insensitive matching
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
        return text

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].apply(_expand_func)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).map_elements(_expand_func, return_dtype=pl.Utf8).alias(column_name)
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
        return df.with_columns(
            df[column_name].cast(pl.Utf8).str.to_lowercase().alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")

def lemmatize(df, column_name):
    """
    Applies lemmatization to words in a text column in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process

    Returns:
        A DataFrame with lemmatized words in the specified column
    """
    try:
        nltk.data.find('tokenizers/wordnet')
    except:
        nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()

    def _lemmatize_func(text):
        if text is None:
            return None
        try:
            words = word_tokenize(str(text))
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        except Exception:
            return str(text)

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].apply(_lemmatize_func)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).map_elements(_lemmatize_func, return_dtype=pl.Utf8).alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")

def stem(df, column_name):
    """
    Applies stemming to words in a text column in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process

    Returns:
        A DataFrame with stemmed words in the specified column
    """
    stemmer = PorterStemmer()

    def _stem_func(text):
        if text is None:
            return None
        try:
            words = word_tokenize(str(text))
            stemmed_words = [stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        except Exception:
            return str(text)

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].apply(_stem_func)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).map_elements(_stem_func, return_dtype=pl.Utf8).alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")

# REMOVING
def remove_stopwords(df, column_name, language='english'):
    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)

    """
    Removes stopwords from a text column in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        language: str - language of stopwords (default: 'english')

    Returns:
        A DataFrame with the specified column having stopwords removed
    """
    stop_words = set(stopwords.words(language))

    # Define the function once
    def _remove_stopwords_func(text):
        if text is None: # Handle potential nulls
            return None
        try:
             # Ensure text is string for tokenization
            words = word_tokenize(str(text))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            return ' '.join(filtered_words)
        except Exception: # Catch potential tokenization errors on weird input
             return str(text) # Return original text if processing fails

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].apply(_remove_stopwords_func)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            df[column_name].map_elements(
                _remove_stopwords_func,
                return_dtype=pl.Utf8 # Specify return type for efficiency
            ).alias(column_name)
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
    space_pattern = r'\s+'

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace(space_pattern, ' ', regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(space_pattern, ' ').alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")

# REPLACING
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
    number_pattern = r'\d+' if not including_floats else  r'-?\b\d+(?:[\.,]\d+)?\b'

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace(number_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(
                number_pattern, replacement
            ).alias(column_name)
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
        df[column_name] = df[column_name].astype(str).str.replace(url_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(
                url_pattern, replacement
            ).alias(column_name)
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
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace(email_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(email_pattern, replacement).alias(column_name)
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
    # Create a regex pattern that matches one or more punctuation characters
    # Use re.escape to handle special characters in string.punctuation correctly
    punct_pattern = f"[{re.escape(string.punctuation)}]+"

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace(punct_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(
                punct_pattern, replacement
            ).alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")

def replace_pattern(df, column_name, regex_pattern, replacement="MATCH"):
    """
    Replaces text matching a regex pattern in a text column of a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        pattern: str - regex pattern to match
        replacement: str - the string to replace matches with (default: 'MATCH')

    Returns:
        A DataFrame with the matched patterns replaced in the specified column
    """
    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace(regex_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(regex_pattern, replacement).alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")
    
def replace_hashtags(df, column_name, replacement="HASHTAG"):
    """
    Replaces hashtags in a text column with a placeholder in a Pandas or Polars DataFrame.

    Args:
        df: A pandas.DataFrame or polars.DataFrame
        column_name: str - name of the column to process
        replacement: str - the string to replace hashtags with (default: 'HASHTAG')

    Returns:
        A DataFrame with hashtags replaced in the specified column
    """
    # Regex to match hashtags: # followed by letters, numbers, or underscores
    hashtag_pattern = r'#[\w]+'

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace(hashtag_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(hashtag_pattern, replacement).alias(column_name)
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
        r'[\U0001F600-\U0001F64F'  # Emoticons
        r'\U0001F300-\U0001F5FF'   # Misc Symbols and Pictographs
        r'\U0001F680-\U0001F6FF'   # Transport and Map Symbols
        r'\U0001F700-\U0001F77F'   # Alchemical Symbols
        r'\U0001F900-\U0001F9FF'   # Supplemental Symbols and Pictographs
        r'\U00002600-\U000026FF'   # Miscellaneous Symbols (e.g., ☀️)
        r'\U00002700-\U000027BF]'  # Dingbats (e.g., ✂️)
    )

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace(emoji_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(emoji_pattern, replacement).alias(column_name)
        )
    else:
        raise TypeError("Unsupported 不过另外一邊是什麼意思 Unsupported DataFrame type. Must be pandas or polars.")

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
    smiley_pattern = r'[:=]-?\)+'

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace(smiley_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(smiley_pattern, replacement).alias(column_name)
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
    currency_pattern = r'[$€£¥₹]\s?-?\d+(?:[\.,]\d+)?|-?\d+(?:[\.,]\d+)?\s?[$€£¥₹]'

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace(currency_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all(currency_pattern, replacement).alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")

def replace_measurements(df, column_name, replacement = "SIZE"):
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
    m_pattern = r'\b\d+(\.\d+)?\s?(oz|kg|g|lb|lbs|m|cm|mm|in|ft|yd|L|ml|gallon|pound)\b'

    if df.__class__.__module__.startswith("pandas"):
        df[column_name] = df[column_name].astype(str).str.replace( m_pattern, replacement, regex=True)
        return df

    elif df.__class__.__module__.startswith("polars"):
        return df.with_columns(
            pl.col(column_name).cast(pl.Utf8).str.replace_all( m_pattern, replacement).alias(column_name)
        )
    else:
        raise TypeError("Unsupported DataFrame type. Must be pandas or polars.")
if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Pandas not installed. Skipping pandas tests.")
        pd = None

    # Updated test data with smiley faces
    test_data = {
        "text": [
            "Here ! is  100 kg 19oz some #SampleText with :) and <USER> 😊 Numbers like 12345",
            "I can't wait for #AwesomeDay :))) with @USER_________ 🎉",
            "You're running at 456.4 mph :-)) see <User> or #FastRun 🚀",
            None,
            "It's $100  a great day 100 =)))))) with #HappyVibes 🌞 at info@site.co.uk"
        ]
    }

    if pd:
        pd_df = pd.DataFrame(test_data)
    if pl:
        pl_df = pl.DataFrame(test_data)

    def run_test(df_type, df):
        print(f"\n{'='*40}\nTesting with {df_type} DataFrame\n{'='*40}")
        orig_df = df.copy() if df_type == "pandas" else df.clone()

        # Existing tests
        print("\n--- Testing lowercase ---")
        print("Before:\n", orig_df["text"])
        result = lowercase(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text")
        print("After:\n", result["text"])

        print("\n--- Testing remove_stopwords ---")
        print("Before:\n", orig_df["text"])
        result = remove_stopwords(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text")
        print("After:\n", result["text"])

        print("\n--- Testing remove_double_spacing ---")
        print("Before:\n", orig_df["text"])
        result = remove_double_spacing(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text")
        print("After:\n", result["text"])

        print("\n--- Testing replace_numbers ---")
        print("Before:\n", orig_df["text"])
        result = replace_numbers(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text", "NUMBER")
        print("After:\n", result["text"])

        print("\n--- Testing replace_urls ---")
        print("Before:\n", orig_df["text"])
        result = replace_urls(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text", "URL_PLACEHOLDER")
        print("After:\n", result["text"])

        print("\n--- Testing replace_punctuation ---")
        print("Before:\n", orig_df["text"])
        result = replace_punctuation(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text", "_PUNCT_")
        print("After:\n", result["text"])

        print("\n--- Testing replace_emails ---")
        print("Before:\n", orig_df["text"])
        result = replace_emails(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text", "EMAIL")
        print("After:\n", result["text"])

        print("\n--- Testing stem_words ---")
        print("Before:\n", orig_df["text"])
        result = stem(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text")
        print("After:\n", result["text"])

        print("\n--- Testing lemmatize_words ---")
        print("Before:\n", orig_df["text"])
        result = lemmatize(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text")
        print("After:\n", result["text"])

        print("\n--- Testing expand_contractions ---")
        print("Before:\n", orig_df["text"])
        result = expand_contractions(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text")
        print("After:\n", result["text"])

        print("\n--- Testing replace_by_regex (for @USER_ patterns) ---")
        print("Before:\n", orig_df["text"])
        user_pattern = r'@USER_+'
        result = replace_pattern(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text", user_pattern, "USER")
        print("After:\n", result["text"])

        print("\n--- Testing replace_hashtags ---")
        print("Before:\n", orig_df["text"])
        result = replace_hashtags(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text", "HASHTAG")
        print("After:\n", result["text"])

        print("\n--- Testing replace_emojis ---")
        print("Before:\n", orig_df["text"])
        result = replace_emojis(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text", "EMOJI")
        print("After:\n", result["text"])

        print("\n--- Testing replace_user_tags ---")
        print("Before:\n", orig_df["text"])
        result = replace_pattern(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text", "USER")
        print("After:\n", result["text"])

        print("\n--- Testing replace_smileys ---")
        print("Before:\n", orig_df["text"])
        result = replace_smileys(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text", "SMILEY")
        print("After:\n", result["text"])

        print("\n--- Testing replace_currency ---")
        print("Before:\n", orig_df["text"])
        result = replace_currency(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text")
        print("After:\n", result["text"])

        print("\n--- Testing replace_measurement ---")
        print("Before:\n", orig_df["text"])
        result = replace_measurements(orig_df.copy() if df_type == "pandas" else orig_df.clone(), "text")
        print("After:\n", result["text"])

    if pd:
        try:
            run_test("pandas", pd_df)
        except Exception as e:
            print(f"\n!!!!!! Error testing pandas functions: {e} !!!!!!\n")

    if pl:
        try:
            run_test("polars", pl_df)
        except Exception as e:
            print(f"\n!!!!!! Error testing polars functions: {e} !!!!!!\n")

    print("\nAll tests completed.")