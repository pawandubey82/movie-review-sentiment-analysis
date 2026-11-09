"""Loading, cleaning and splitting the IMDb dataset."""
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV file."""
    return pd.read_csv(path)


def clean_review(text: str) -> str:
    """Remove HTML line breaks and extra whitespace."""
    text = str(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing/duplicate rows, clean text, encode labels, add length features."""
    print("Missing values:\n", df.isnull().sum(), sep="")
    print("Duplicate reviews:", df["review"].duplicated().sum())

    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[df["sentiment"].isin(config.LABEL_MAP)].copy()
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

    df["review"] = df["review"].apply(clean_review)
    df["label"] = df["sentiment"].map(config.LABEL_MAP)
    df["word_count"] = df["review"].apply(lambda x: len(x.split()))
    df["char_count"] = df["review"].str.len()

    print("Cleaned dataset shape:", df.shape)
    return df


def get_data(sample: int | None = None) -> pd.DataFrame:
    """Load + preprocess. `sample` keeps only N rows (handy for quick tests)."""
    df = preprocess(load_raw_data())
    if sample:
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
        print(f"Using a random sample of {sample} reviews")
    return df


def split_data(df: pd.DataFrame):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    return train_test_split(
        df["review"],
        df["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )
