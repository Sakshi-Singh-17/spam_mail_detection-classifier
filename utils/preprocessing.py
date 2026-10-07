"""
utils/preprocessing.py
-----------------------
Shared text preprocessing utilities used by both train_model.py and app.py.

Keeping preprocessing logic in one place ensures the EXACT same transformations
are applied at training time and at inference time — a critical requirement to
avoid training/serving skew.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Apply lightweight, spam-aware text cleaning to a single email string.

    Design decisions:
    - We do NOT strip URLs, numbers, or punctuation aggressively because
      tokens like "http", "free!!", "$1000", "click here" are strong spam
      signals that TF-IDF should capture.
    - We normalise whitespace so the vectoriser tokenises cleanly.
    - We lower-case so "FREE" and "free" map to the same token.

    Parameters
    ----------
    text : str
        Raw email / message text.

    Returns
    -------
    str
        Cleaned text ready for TF-IDF vectorisation.
    """
    # Coerce to string in case of unexpected numeric types
    text = str(text)

    # Lowercase — standard normalisation
    text = text.lower()

    # Collapse repeated whitespace characters (tabs, newlines, carriage returns)
    # into a single space so the tokeniser sees clean word boundaries
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def preprocess_dataframe(df, text_col: str, label_col: str):
    """
    Clean a raw spam/ham DataFrame and return a tidy copy with two columns:
    'text' (cleaned) and 'label' (integer: 1 = spam, 0 = ham).

    Deduplication is performed on the cleaned text so that exact or near-exact
    duplicate emails (known to exist in this corpus) do not inflate evaluation
    metrics by appearing in both train and test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    text_col : str
        Name of the column containing email text.
    label_col : str
        Name of the column containing the numeric label (1 = spam, 0 = ham).

    Returns
    -------
    pd.DataFrame
        Cleaned, deduplicated DataFrame with columns ['text', 'label'].
    """
    import pandas as pd

    out = pd.DataFrame()

    # Apply text cleaning
    out["text"] = df[text_col].apply(clean_text)

    # Keep the integer label as-is (already 0/1 in this dataset)
    out["label"] = df[label_col].astype(int)

    # Drop rows with empty text after cleaning
    out = out[out["text"].str.len() > 0]

    # Drop rows with missing labels
    out = out.dropna(subset=["label"])

    # Remove exact duplicate texts.
    # Duplicates in this Enron-based corpus can appear in both the train and
    # test splits, which inflates accuracy/F1 scores.  We keep the first
    # occurrence (preserving label distribution order).
    before = len(out)
    out = out.drop_duplicates(subset="text", keep="first")
    after = len(out)
    if before != after:
        import warnings
        warnings.warn(
            f"Dropped {before - after} duplicate rows (exact text match) "
            f"before train/test split.",
            stacklevel=2,
        )

    # Reset index for a clean sequential index
    out = out.reset_index(drop=True)

    return out
