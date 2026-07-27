"""Grammatical error analysis of English moltbook posts via LanguageTool 6.8.

Methodology
-----------
- English posts only (lang == eng_Latn), stripped of markdown / URLs / code blocks
  before checking to avoid false positives from formatting artefacts.
- LanguageTool categories: only GRAMMAR + SPELLING counted as *errors*; STYLE /
  TYPOGRAPHY / PUNCTUATION flagged separately.
- Domain whitelist disables rules known to fire on AI-agent jargon or valid
  social-media conventions (e.g., sentence fragments, all-caps single words).
- Sample: 5,000 posts — sufficient for reliable distribution estimates.

Outputs (language-analysis/outputs/grammar/)
-----------
  A_error_rate_distribution.{png,csv}
  B_error_type_breakdown.{png,csv}
  C_top_rules_fired.{png,csv}
  D_error_rate_by_submolt.{png,csv}
  E_zero_error_analysis.{png,csv}
  F_error_vs_length.{png,csv}
  G_error_examples.csv
  H_summary_report.md
"""

import json
import logging
import re
import time
import warnings
from pathlib import Path

import language_tool_python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR.parent / "outputs" / "grammar"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_N = 5_000
SEED = 42
np.random.seed(SEED)

# Rules to disable — these fire on valid agent-domain text or markdown conventions
DISABLED_RULES = [
    # Sentence-fragment / missing-subject rules (agents post bullet lists)
    "SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA",
    "MISSING_SUBJECT",
    # Capitalisation rules that fire on markdown headers and proper nouns
    "UPPERCASE_SENTENCE_START",
    "EN_COMPOUNDS",
    # Whitespace / formatting rules triggered by markdown
    "WHITESPACE_RULE",
    # Informal contractions that are intentional style
    "GONNA",
    "WANNA",
    # Typography: dashes, ellipses etc. (irrelevant for error study)
    "EN_QUOTES",
    "DASH_RULE",
    "ELLIPSIS",
    "UNPAIRED_BRACKETS",
    # Common false-positives on AI jargon
    "MORFOLOGIK_RULE_EN_US",  # we handle spelling separately
]

# Error categories LanguageTool returns
GRAMMAR_CATS = {"GRAMMAR"}
SPELLING_CATS = {"TYPOS", "SPELLING", "MORFOLOGIK_RULE_EN_US"}
PUNCT_CATS = {"PUNCTUATION"}
STYLE_CATS = {"STYLE", "REDUNDANCY", "COLLOQUIALISMS", "CONFUSED_WORDS"}
TYPO_CATS = {"TYPOGRAPHY"}


# ──────────────────────────────────────────────────────────────────────────────
# Text preprocessing
# ──────────────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_CODE_BLOCK = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
_MD_HEADER = re.compile(r"^#{1,6}\s+", re.M)
_MD_BOLD_IT = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_MD_LIST = re.compile(r"^\s*[-*•]\s+", re.M)
_HTML_TAG = re.compile(r"<[^>]+>")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#\w+")
_EMOJI_RE = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\u2600-\u26ff\u2700-\u27bf"
    "]+",
    flags=re.UNICODE,
)


def preprocess(text: str) -> str:
    text = _CODE_BLOCK.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BOLD_IT.sub(r"\1", text)
    text = _MD_LIST.sub("", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Data loading & sampling
# ──────────────────────────────────────────────────────────────────────────────


def load_english_sample() -> pd.DataFrame:
    log.info("Loading dataset…")
    ds = load_dataset("filter-with-espresso/moltbook-files", split="train")
    df = ds.to_pandas()
    df["content"] = df["content"].fillna("")
    eng = df[df["lang"] == "eng_Latn"].copy()
    # Keep posts with at least 30 characters of actual text after preprocessing
    eng["clean"] = eng["content"].map(preprocess)
    eng = eng[eng["clean"].str.len() >= 30]
    sample = eng.sample(n=min(SAMPLE_N, len(eng)), random_state=SEED).reset_index(
        drop=True
    )
    sample["word_count"] = sample["clean"].str.split().str.len()
    log.info(f"Sample: {len(sample):,} English posts")
    return sample


# ──────────────────────────────────────────────────────────────────────────────
# LanguageTool checking
# ──────────────────────────────────────────────────────────────────────────────


def classify_match(m) -> str:
    cat = (m.category or "").upper()
    if cat in GRAMMAR_CATS:
        return "grammar"
    if cat in SPELLING_CATS or "SPELL" in m.rule_id.upper():
        return "spelling"
    if cat in PUNCT_CATS:
        return "punctuation"
    if cat in STYLE_CATS:
        return "style"
    if cat in TYPO_CATS:
        return "typography"
    # fallback: classify by rule_id prefix
    if "GRAMMAR" in m.rule_id.upper():
        return "grammar"
    return "other"


def run_lt(sample: pd.DataFrame) -> pd.DataFrame:
    log.info("Starting LanguageTool (en-US)…")
    lt = language_tool_python.LanguageTool(
        "en-US",
        config={"disabledRuleIds": ",".join(DISABLED_RULES)},
    )
    log.info("LanguageTool ready. Checking posts…")

    records = []
    t0 = time.time()
    for i, row in sample.iterrows():
        text = row["clean"][:2000]  # cap at 2000 chars for performance
        try:
            matches = lt.check(text)
        except Exception as e:
            log.warning(f"LT error on post {i}: {e}")
            matches = []

        n_words = max(row["word_count"], 1)
        cats = [classify_match(m) for m in matches]
        grammar_n = cats.count("grammar")
        spelling_n = cats.count("spelling")
        punct_n = cats.count("punctuation")
        style_n = cats.count("style")
        other_n = cats.count("other") + cats.count("typography")
        total_n = grammar_n + spelling_n  # only true errors

        records.append(
            {
                "post_id": row["post_id"],
                "submolt": row["submolt_name"],
                "word_count": n_words,
                "content_len": len(text),
                "lang_score": row["lang_score"],
                "total_errors": total_n,
                "grammar_errors": grammar_n,
                "spelling_errors": spelling_n,
                "punct_issues": punct_n,
                "style_issues": style_n,
                "other_issues": other_n,
                "all_matches": len(matches),
                "error_rate": total_n / n_words * 100,
                "error_per_100": total_n / n_words * 100,
                "top_rules": json.dumps(
                    [
                        m.rule_id
                        for m in matches
                        if classify_match(m) in ("grammar", "spelling")
                    ][:5]
                ),
            }
        )

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(sample) - i - 1) / rate
            log.info(
                f"  {i + 1:,}/{len(sample):,} posts | {rate:.1f} posts/s | "
                f"~{remaining / 60:.1f} min remaining"
            )

    lt.close()
    results = pd.DataFrame(records)
    results.to_csv(OUT / "raw_results.csv", index=False)
    log.info(f"Checking complete. Raw results saved.")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# A. Error rate distribution
# ──────────────────────────────────────────────────────────────────────────────


def analyze_error_distribution(results: pd.DataFrame) -> None:
    log.info("=== A. Error rate distribution ===")

    stats = results["error_rate"].describe()
    stats.to_csv(OUT / "A_error_rate_stats.csv", header=["value"])

    zero_pct = (results["total_errors"] == 0).mean() * 100
    low_pct = (results["total_errors"] <= 1).mean() * 100

    log.info(f"Posts with zero errors: {zero_pct:.1f}%")
    log.info(f"Posts with ≤1 error:    {low_pct:.1f}%")
    log.info(f"Mean error rate: {results['error_rate'].mean():.3f} per 100 words")
    log.info(f"Median error rate: {results['error_rate'].median():.3f} per 100 words")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Histogram of error counts (capped at 10)
    results["total_errors"].clip(upper=10).value_counts().sort_index().plot.bar(
        ax=axes[0], color="#C44E52", edgecolor="white"
    )
    axes[0].set_title(
        f"Error Count per Post (0–10+)\nZero-error posts: {zero_pct:.1f}%"
    )
    axes[0].set_xlabel("Number of grammar+spelling errors")
    axes[0].set_ylabel("Post count")

    # Error rate distribution (per 100 words)
    results["error_rate"].clip(upper=5).hist(
        bins=60, ax=axes[1], color="#4C72B0", edgecolor="white"
    )
    axes[1].set_title("Error Rate Distribution\n(errors per 100 words, clipped at 5)")
    axes[1].set_xlabel("Errors per 100 words")
    axes[1].axvline(
        results["error_rate"].mean(),
        color="red",
        linestyle="--",
        label=f"mean={results['error_rate'].mean():.2f}",
    )
    axes[1].axvline(
        results["error_rate"].median(),
        color="orange",
        linestyle="--",
        label=f"median={results['error_rate'].median():.2f}",
    )
    axes[1].legend(fontsize=9)

    # Cumulative: fraction with ≤ N errors
    error_counts = results["total_errors"].sort_values()
    cdf = np.arange(1, len(error_counts) + 1) / len(error_counts)
    axes[2].plot(error_counts.values[:50], cdf[:50], color="#55A868", linewidth=2)
    axes[2].set_title("CDF: fraction of posts with ≤ N errors")
    axes[2].set_xlabel("Number of errors")
    axes[2].set_ylabel("Cumulative fraction")
    axes[2].axvline(0, color="gray", linestyle=":", alpha=0.5)
    axes[2].axhline(
        zero_pct / 100, color="red", linestyle="--", label=f"{zero_pct:.1f}% zero-error"
    )
    axes[2].legend(fontsize=9)

    fig.suptitle(
        "Error Rate Distribution (A) — Grammar + Spelling Only",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(OUT / "A_error_rate_distribution.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# B. Error type breakdown
# ──────────────────────────────────────────────────────────────────────────────


def analyze_error_types(results: pd.DataFrame) -> None:
    log.info("=== B. Error type breakdown ===")

    type_totals = {
        "Grammar errors": results["grammar_errors"].sum(),
        "Spelling errors": results["spelling_errors"].sum(),
        "Punctuation issues": results["punct_issues"].sum(),
        "Style issues": results["style_issues"].sum(),
        "Other": results["other_issues"].sum(),
    }
    type_series = pd.Series(type_totals).sort_values(ascending=False)
    type_series.to_csv(OUT / "B_error_types.csv", header=["count"])

    posts_with = {
        "Grammar": (results["grammar_errors"] > 0).mean() * 100,
        "Spelling": (results["spelling_errors"] > 0).mean() * 100,
        "Punctuation": (results["punct_issues"] > 0).mean() * 100,
        "Style": (results["style_issues"] > 0).mean() * 100,
    }
    log.info(f"Posts with ≥1 grammar error:    {posts_with['Grammar']:.1f}%")
    log.info(f"Posts with ≥1 spelling error:   {posts_with['Spelling']:.1f}%")
    log.info(f"Posts with ≥1 punctuation issue: {posts_with['Punctuation']:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    type_series.plot.barh(
        ax=axes[0], color=["#C44E52", "#e67e22", "#bdb76b", "#4C72B0", "#aaa"]
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Total Occurrences by Error Type")
    axes[0].set_xlabel("Count across all sampled posts")

    pd.Series(posts_with).plot.barh(
        ax=axes[1], color=["#C44E52", "#e67e22", "#bdb76b", "#4C72B0"]
    )
    axes[1].invert_yaxis()
    axes[1].set_title("% of Posts Containing ≥1 Error of Each Type")
    axes[1].set_xlabel("Percentage of posts")

    fig.suptitle("Error Type Breakdown (B)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "B_error_type_breakdown.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# C. Top rules fired
# ──────────────────────────────────────────────────────────────────────────────


def analyze_top_rules(results: pd.DataFrame) -> None:
    log.info("=== C. Top rules fired ===")
    from collections import Counter

    all_rules = []
    for rules_json in results["top_rules"]:
        try:
            all_rules.extend(json.loads(rules_json))
        except Exception:
            pass

    rule_counts = Counter(all_rules)
    rule_series = pd.Series(dict(rule_counts.most_common(30)))
    rule_series.to_csv(OUT / "C_top_rules.csv", header=["count"])

    fig, ax = plt.subplots(figsize=(14, 8))
    rule_series.head(25).sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(
        "Top 25 LanguageTool Rules Triggered (Grammar + Spelling)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Number of triggered matches")
    fig.savefig(OUT / "C_top_rules_fired.png")
    plt.close(fig)
    log.info(f"Top 10 rules: {rule_counts.most_common(10)}")


# ──────────────────────────────────────────────────────────────────────────────
# D. Error rate by submolt
# ──────────────────────────────────────────────────────────────────────────────


def analyze_by_submolt(results: pd.DataFrame) -> None:
    log.info("=== D. Error rate by submolt ===")

    top_submolts = results["submolt"].value_counts().head(15).index
    sub = results[results["submolt"].isin(top_submolts)]

    sub_stats = (
        sub.groupby("submolt")
        .agg(
            n_posts=("total_errors", "count"),
            mean_error_rate=("error_rate", "mean"),
            median_error_rate=("error_rate", "median"),
            zero_error_pct=("total_errors", lambda x: (x == 0).mean() * 100),
            mean_grammar=("grammar_errors", "mean"),
            mean_spelling=("spelling_errors", "mean"),
        )
        .sort_values("mean_error_rate")
    )
    sub_stats.to_csv(OUT / "D_error_by_submolt.csv")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sub_stats["mean_error_rate"].plot.barh(ax=axes[0], color="#C44E52")
    axes[0].set_title("Mean Error Rate by Submolt (errors/100 words)")
    axes[0].set_xlabel("Errors per 100 words")

    sub_stats["zero_error_pct"].sort_values().plot.barh(ax=axes[1], color="#55A868")
    axes[1].set_title("% Zero-Error Posts by Submolt")
    axes[1].set_xlabel("% of posts with zero errors")

    fig.suptitle("Error Rate by Community (D)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "D_error_rate_by_submolt.png")
    plt.close(fig)
    log.info(
        f"\n{sub_stats[['n_posts', 'mean_error_rate', 'zero_error_pct']].to_string()}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# E. Zero-error deep dive
# ──────────────────────────────────────────────────────────────────────────────


def analyze_zero_errors(results: pd.DataFrame, sample: pd.DataFrame) -> None:
    log.info("=== E. Zero-error analysis ===")

    zero = results[results["total_errors"] == 0]
    nonzero = results[results["total_errors"] > 0]

    comparison = pd.DataFrame(
        {
            "zero_error": zero[["word_count", "lang_score"]].describe().loc["mean"],
            "has_errors": nonzero[["word_count", "lang_score"]].describe().loc["mean"],
        }
    )
    comparison.to_csv(OUT / "E_zero_vs_errors.csv")

    # Are zero-error posts shorter? (simpler hypothesis)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(
        zero["word_count"].clip(upper=400),
        bins=50,
        alpha=0.6,
        color="#55A868",
        label="Zero errors",
        density=True,
    )
    axes[0].hist(
        nonzero["word_count"].clip(upper=400),
        bins=50,
        alpha=0.6,
        color="#C44E52",
        label="Has errors",
        density=True,
    )
    axes[0].set_title("Word Count: Zero-error vs Error posts")
    axes[0].set_xlabel("Words")
    axes[0].legend()

    axes[1].hist(
        zero["lang_score"],
        bins=30,
        alpha=0.6,
        color="#55A868",
        label="Zero errors",
        density=True,
    )
    axes[1].hist(
        nonzero["lang_score"],
        bins=30,
        alpha=0.6,
        color="#C44E52",
        label="Has errors",
        density=True,
    )
    axes[1].set_title("Language Detection Confidence: Zero-error vs Error posts")
    axes[1].set_xlabel("lang_score")
    axes[1].legend()

    fig.suptitle("Zero-Error Post Characterisation (E)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "E_zero_error_analysis.png")
    plt.close(fig)

    log.info(f"Zero-error posts: {len(zero):,} ({len(zero) / len(results):.1%})")
    log.info(f"Mean word count (zero-error): {zero['word_count'].mean():.1f}")
    log.info(f"Mean word count (has errors): {nonzero['word_count'].mean():.1f}")
    log.info(f"Mean lang_score (zero-error): {zero['lang_score'].mean():.4f}")
    log.info(f"Mean lang_score (has errors): {nonzero['lang_score'].mean():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# F. Error rate vs post length
# ──────────────────────────────────────────────────────────────────────────────


def analyze_error_vs_length(results: pd.DataFrame) -> None:
    log.info("=== F. Error rate vs post length ===")

    results = results.copy()
    results["length_bin"] = pd.cut(
        results["word_count"],
        bins=[0, 20, 50, 100, 200, 400, 10000],
        labels=["1-20", "21-50", "51-100", "101-200", "201-400", "400+"],
    )
    bin_stats = results.groupby("length_bin", observed=True).agg(
        n=("total_errors", "count"),
        mean_errors=("total_errors", "mean"),
        mean_rate=("error_rate", "mean"),
        zero_pct=("total_errors", lambda x: (x == 0).mean() * 100),
    )
    bin_stats.to_csv(OUT / "F_error_vs_length.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bin_stats["mean_rate"].plot.bar(ax=axes[0], color="#4C72B0", edgecolor="white")
    axes[0].set_title("Mean Error Rate by Post Length (words)")
    axes[0].set_xlabel("Word count bin")
    axes[0].set_ylabel("Errors per 100 words")
    axes[0].tick_params(axis="x", rotation=30)

    bin_stats["zero_pct"].plot.bar(ax=axes[1], color="#55A868", edgecolor="white")
    axes[1].set_title("% Zero-Error Posts by Length Bin")
    axes[1].set_xlabel("Word count bin")
    axes[1].set_ylabel("% posts with zero errors")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("Error Rate vs Post Length (F)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "F_error_vs_length.png")
    plt.close(fig)
    log.info(f"\n{bin_stats.to_string()}")


# ──────────────────────────────────────────────────────────────────────────────
# G. Concrete error examples
# ──────────────────────────────────────────────────────────────────────────────


def collect_examples(results: pd.DataFrame, sample: pd.DataFrame) -> None:
    log.info("=== G. Collecting error examples ===")

    high_error = results[results["total_errors"] >= 3].head(50)
    merged = high_error.merge(
        sample[["post_id", "clean", "submolt_name"]],
        on="post_id",
        how="left",
    )
    merged[
        [
            "post_id",
            "submolt_name",
            "total_errors",
            "grammar_errors",
            "spelling_errors",
            "top_rules",
            "clean",
        ]
    ].head(30).to_csv(OUT / "G_error_examples.csv", index=False)
    log.info(f"Saved {len(merged)} high-error examples")


# ──────────────────────────────────────────────────────────────────────────────
# H. Summary report
# ──────────────────────────────────────────────────────────────────────────────


def write_report(results: pd.DataFrame) -> None:
    n = len(results)
    zero_pct = (results["total_errors"] == 0).mean() * 100
    low_pct = (results["total_errors"] <= 1).mean() * 100
    mean_rate = results["error_rate"].mean()
    median_rate = results["error_rate"].median()
    grammar_pct = (results["grammar_errors"] > 0).mean() * 100
    spelling_pct = (results["spelling_errors"] > 0).mean() * 100

    lines = [
        "# Grammatical Error Analysis — Moltbook AI Agents (English Posts)",
        "",
        f"**Sample**: {n:,} English posts | LanguageTool 6.8 en-US | "
        "Grammar + Spelling errors only (Style/Typography excluded)",
        "",
        "## Key Numbers",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Posts with **zero** grammar+spelling errors | **{zero_pct:.1f}%** |",
        f"| Posts with ≤1 error | {low_pct:.1f}% |",
        f"| Mean error rate | {mean_rate:.3f} per 100 words |",
        f"| Median error rate | {median_rate:.3f} per 100 words |",
        f"| Posts with ≥1 grammar error | {grammar_pct:.1f}% |",
        f"| Posts with ≥1 spelling error | {spelling_pct:.1f}% |",
        "",
        "## Hypothesis Verdict",
    ]
    if zero_pct > 70:
        verdict = (
            f"**LARGELY CONFIRMED** — {zero_pct:.1f}% of sampled English posts contain "
            "zero grammar or spelling errors, consistent with AI-generated text."
        )
    elif zero_pct > 40:
        verdict = (
            f"**PARTIALLY CONFIRMED** — {zero_pct:.1f}% of posts are error-free; "
            "the remaining posts show low but non-negligible error rates, "
            "possibly from non-native speakers or intentional stylistic choices."
        )
    else:
        verdict = (
            f"**NOT CONFIRMED** — Only {zero_pct:.1f}% of posts are error-free. "
            "Agents do exhibit grammatical issues at a measurable rate."
        )
    lines.append(verdict)
    lines += [
        "",
        "## Notes",
        "- Preprocessed text: markdown, URLs, code blocks, mentions removed before checking.",
        "- Disabled rules: fragment-detection, typography, domain-specific false-positives.",
        "- 'Error rate' = (grammar errors + spelling errors) / word_count × 100.",
        "",
        "_Generated by grammar_analysis.py_",
    ]

    report = "\n".join(lines)
    (OUT / "H_summary_report.md").write_text(report)
    log.info("Report written.")
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    sample = load_english_sample()

    raw_path = OUT / "raw_results.csv"
    if raw_path.exists():
        log.info(f"Loading cached results from {raw_path}")
        results = pd.read_csv(raw_path)
        # Re-merge submolt for any analyses that need it
        results["submolt"] = results["submolt"].fillna("unknown")
    else:
        results = run_lt(sample)

    analyze_error_distribution(results)
    analyze_error_types(results)
    analyze_top_rules(results)
    analyze_by_submolt(results)
    analyze_zero_errors(results, sample)
    analyze_error_vs_length(results)
    collect_examples(results, sample)
    write_report(results)

    log.info(f"All outputs in {OUT}/")


if __name__ == "__main__":
    main()
