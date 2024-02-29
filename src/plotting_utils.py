from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import set_matplotlib_formats
from sklearn.feature_extraction.text import CountVectorizer

set_matplotlib_formats("retina")


def plot_per_type(
    dataframe: pd.DataFrame,
    column: str,
    title: str,
    title_shift: int = 1.1,
    bins: int = 10,
    figsize: Tuple[int] = (8, 4),
):
    f, ax = plt.subplots(1, 2, figsize=figsize)

    dataframe[dataframe["type"] == "movie"][column].plot.hist(
        ax=ax[0], edgecolor="black", color="red", bins=bins
    )
    ax[0].set_title("Type = movie")

    dataframe[dataframe["type"] == "series"][column].plot.hist(
        ax=ax[1], edgecolor="black", color="green", bins=bins
    )
    ax[1].set_title("Type = series")

    f.tight_layout()

    f.suptitle(title, y=title_shift)
    plt.show()


def plot_dt_per_type(
    dataframe: pd.DataFrame,
    column: str,
    title: str,
    title_shift: int = 1.0,
    bins: int = 24,
    figsize: Tuple[int] = (14, 4),
):
    f, ax = plt.subplots(1, 2, figsize=figsize)

    dataframe[dataframe["type"] == "movie"][column].hist(
        ax=ax[0], edgecolor="black", color="red", bins=np.arange(bins + 1) - 0.5
    )
    ax[0].set_title("Type = movie")
    ax[0].set_xticks(range(bins), minor=False)
    ax[0].grid(visible=None)

    dataframe[dataframe["type"] == "series"][column].hist(
        ax=ax[1], edgecolor="black", color="green", bins=np.arange(bins + 1) - 0.5
    )
    ax[1].set_title("Type = series")
    ax[1].set_xticks(range(bins), minor=False)
    ax[1].grid(visible=None)

    f.suptitle(title, y=title_shift)

    plt.show()


def plot_catplot(
    y: str,
    x: str,
    hue: str,
    data: pd.DataFrame,
    title: str,
    kind: str = "box",
    order: List[str] = ["negative", "positive"],
    medianprops: dict = {"color": "red", "lw": 1},
):
    if kind == "box":
        plot = sns.catplot(
            y=y,
            x=x,
            hue=hue,
            data=data,
            kind=kind,
            order=order,
            medianprops=medianprops,
        )
    else:
        plot = sns.catplot(y=y, x=x, hue=hue, data=data, kind=kind, order=order)

    plot.figure.subplots_adjust(top=0.9)
    plot.fig.suptitle(title)
    plt.show()


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel("Length of a sample")
    plt.ylabel("Number of samples")
    plt.title("Sample length distribution")
    plt.show()


def plot_frequency_distribution_of_ngrams(
    sample_texts, ngram_range=(1, 1), num_ngrams=50
):
    """Plots the frequency distribution of n-grams.

    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, max), The range of n-gram values to consider.
            min and max are the lower and the upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
        "ngram_range": ngram_range,
        "dtype": "int32",
        "strip_accents": "unicode",
        "decode_error": "replace",
        "analyzer": "word",  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # indices). This also converts every text to an array the length of
    # vocabulary, where every element represents the count of the n-gram
    # corresponding at that index in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names_out())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(
        *[(c, n) for c, n in sorted(zip(all_counts, all_ngrams), reverse=True)]
    )
    ngrams = list(all_ngrams)[:num_ngrams][::-1]
    counts = list(all_counts)[:num_ngrams][::-1]

    idx = np.arange(num_ngrams)

    f, ax = plt.subplots(figsize=(5, 8))
    plt.barh(idx, counts, align="center")
    plt.ylabel("Top {num_ngrams} N-grams".format(num_ngrams=num_ngrams))
    plt.xlabel("Frequencies")
    plt.title(
        "Frequency distribution of n-grams with range={ngram_range}".format(
            ngram_range=ngram_range
        )
    )
    plt.yticks(idx, ngrams)
    plt.margins(0.05, 0.01)
    plt.show()
