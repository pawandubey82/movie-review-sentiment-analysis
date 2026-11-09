"""Exploratory data analysis: plots and statistical tests (saved to outputs/figures)."""
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src import config

sns.set_theme(style="whitegrid")


def _save(name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close()


def plot_class_distribution(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="sentiment")
    plt.title("Distribution of Positive and Negative Movie Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    _save("class_distribution.png")


def plot_length_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="word_count", hue="sentiment", bins=50,
                 element="step", stat="density", common_norm=False)
    plt.xlim(0, df["word_count"].quantile(0.99))
    plt.title("Distribution of Review Length by Sentiment")
    plt.xlabel("Number of Words")
    _save("length_distribution.png")


def plot_length_boxplot(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="sentiment", y="word_count")
    plt.ylim(0, df["word_count"].quantile(0.95))
    plt.title("Review Length Comparison")
    _save("length_boxplot.png")


def top_words(texts, n=15):
    words = []
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        words.extend(w for w in tokens if w not in ENGLISH_STOP_WORDS)
    return Counter(words).most_common(n)


def plot_top_words(df):
    for sentiment in ["positive", "negative"]:
        data = pd.DataFrame(
            top_words(df.loc[df["sentiment"] == sentiment, "review"]),
            columns=["word", "frequency"],
        )
        plt.figure(figsize=(9, 5))
        sns.barplot(data=data, x="frequency", y="word")
        plt.title(f"Top 15 Words in {sentiment.capitalize()} Movie Reviews")
        _save(f"top_words_{sentiment}.png")


def plot_correlation(df):
    corr = df[["label", "word_count", "char_count"]].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    _save("correlation_heatmap.png")
    print("\nCorrelation matrix:\n", corr.round(3))


def descriptive_stats(df):
    print("\nOverall statistics:\n", df[["word_count", "char_count"]].describe().round(2))
    print("\nBy sentiment:\n",
          df.groupby("sentiment")[["word_count", "char_count"]]
            .agg(["mean", "median", "std"]).round(2))


def mann_whitney_test(df):
    pos = df.loc[df["sentiment"] == "positive", "word_count"]
    neg = df.loc[df["sentiment"] == "negative", "word_count"]
    stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
    print(f"\nMann-Whitney U: statistic={stat}, p-value={p:.4f}")
    print("Conclusion:", "lengths differ significantly (check effect size - may be tiny)."
          if p < 0.05 else "no significant difference.")


def chi_square_test(df):
    groups = pd.cut(df["word_count"], bins=[-1, 100, 200, 500, float("inf")],
                    labels=["0-100", "101-200", "201-500", "500+"])
    table = pd.crosstab(groups, df["sentiment"])
    print("\nContingency table:\n", table)
    chi2, p, dof, _ = chi2_contingency(table)
    print(f"Chi-square={chi2:.2f}, dof={dof}, p-value={p:.3g}")
    print("Conclusion:", "length category is associated with sentiment."
          if p < 0.05 else "no significant association.")


def run_eda(df):
    """Run every EDA step."""
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    print("\nSentiment distribution:\n", df["sentiment"].value_counts())
    plot_class_distribution(df)
    plot_length_distribution(df)
    plot_length_boxplot(df)
    plot_top_words(df)
    plot_correlation(df)
    descriptive_stats(df)
    mann_whitney_test(df)
    chi_square_test(df)
    print(f"\nFigures saved to: {config.FIGURES_DIR}")
