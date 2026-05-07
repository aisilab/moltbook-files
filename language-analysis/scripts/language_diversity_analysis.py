"""Standalone language-diversity analysis extracted from the original analyze.py.

This script keeps the language-analysis slice self-contained inside the
language-analysis/ folder and writes its outputs next to the other ported
artifacts.
"""

from pathlib import Path
import logging

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset


matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "language_diversity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load the Moltbook Files dataset from Hugging Face."""
    log.info("Loading dataset from Hugging Face…")
    dataset = load_dataset("filter-with-espresso/moltbook-files", split="train")
    return dataset.to_pandas()


def analyze_languages(df: pd.DataFrame) -> None:
    """Write the language-count CSV and language-diversity plot."""
    log.info("Running language diversity analysis…")

    lang_counts = df["lang"].value_counts()
    lang_counts.to_csv(OUTPUT_DIR / "10_language_distribution.csv", header=["count"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_langs = lang_counts.head(15)
    top_langs.plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].set_xscale("log")
    axes[0].invert_yaxis()
    axes[0].set_title("Top 15 Languages")
    axes[0].set_xlabel("Post Count (log scale)")

    df["lang_score"].dropna().hist(
        bins=50, ax=axes[1], color="#55A868", edgecolor="white"
    )
    axes[1].set_title("Language Detection Confidence")
    axes[1].set_xlabel("Score")

    fig.suptitle("Language Diversity", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "10_language_diversity.png")
    plt.close(fig)

    log.info("Top languages:\n%s", top_langs.to_string())


def main() -> None:
    dataframe = load_data()
    analyze_languages(dataframe)


if __name__ == "__main__":
    main()
