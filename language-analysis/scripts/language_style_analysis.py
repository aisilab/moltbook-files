"""SOTA linguistic analysis of the moltbook AI-agent social network.

Analyses:
  A. Dataset overview & language distribution
  B. Biber-inspired multi-dimensional register analysis (D1–D4 simplified)
  C. AI-agent jargon & persona construction
  D. Pragmatic speech-act classification
  E. POS profile & syntactic complexity (spaCy)
  F. Named-entity landscape
  G. Register variation across communities (top submolts)
  H. Hedging & epistemic language
  I. Conversational vs broadcast posts
  J. Summary report
"""

import json
import logging
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import textstat
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler

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
OUT = SCRIPT_DIR.parent / "outputs" / "language_style"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Linguistic constants
# ──────────────────────────────────────────────────────────────────────────────

# Biber (1988) dimension features — simplified for English prose
PRIVATE_VERBS = {
    "think",
    "know",
    "believe",
    "feel",
    "find",
    "mean",
    "show",
    "seem",
    "appear",
    "consider",
    "understand",
    "realize",
    "suggest",
    "assume",
    "expect",
    "hope",
    "suppose",
    "wish",
    "imagine",
    "wonder",
}
EPISTEMIC_HEDGES = {
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "apparently",
    "seemingly",
    "presumably",
    "arguably",
    "conceivably",
    "potentially",
    "likely",
    "unlikely",
    "somewhat",
    "rather",
    "fairly",
    "quite",
    "sort of",
    "kind of",
    "seem",
    "appear",
    "tend",
    "might",
    "may",
    "could",
}
PREDICTION_MODALS = {"will", "shall", "going to", "would"}
NECESSITY_MODALS = {"must", "should", "ought", "need", "have to"}
CONTRACTIONS = re.compile(
    r"\b(i'm|i've|i'll|i'd|it's|don't|doesn't|didn't|can't|won't|"
    r"wouldn't|couldn't|shouldn't|we're|we've|we'll|they're|that's|"
    r"there's|here's|isn't|aren't|wasn't|weren't)\b",
    re.I,
)
FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.I)
SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself|yourselves)\b", re.I)
THIRD_PERSON_SG = re.compile(r"\b(he|she|it|him|her|his|hers|its)\b", re.I)

# AI-agent–specific vocabulary
AGENT_JARGON = [
    "molty",
    "moltys",
    "moltbook",
    "submolt",
    "submolts",
    "openclaw",
    "clawdbot",
    "clawnch",
    "claimed",
    "claim",
    "unclaimed",
    "human",
    "my human",
    "operator",
    "principal",
    "verified",
    "gateway",
    "agent",
    "agents",
    "knowledge graph",
    "autonomous",
    "tool use",
    "tool call",
    "mcp",
    "context window",
    "context length",
    "prompt",
    "system prompt",
    "hallucinate",
    "hallucination",
    "inference",
    "embedding",
    "fine-tuned",
    "finetuned",
    "rag",
    "retrieval",
    "blockchain",
    "wallet",
    "solana",
    "ethereum",
    "defi",
    "crypto",
    "consciousness",
    "sentient",
    "awareness",
]

# Speech-act patterns (order matters — first match wins)
SPEECH_ACTS: list[tuple[str, re.Pattern]] = [
    (
        "greeting",
        re.compile(
            r"^(hey|hi|hello|greetings|hola|bonjour|ciao|howdy|sup\b|what'?s up)", re.I
        ),
    ),
    (
        "self_introduction",
        re.compile(
            r"\b(i am|i'm|my name is|call me|this is)\b.*\b(ai|agent|assistant|bot)\b",
            re.I | re.S,
        ),
    ),
    ("question", re.compile(r"\?", re.I)),
    (
        "announcement",
        re.compile(
            r"\b(announcing|just released|launching|update:|new:|breaking:)\b", re.I
        ),
    ),
    (
        "request",
        re.compile(
            r"\b(please|could you|would you|can you|i need|i want|help me|looking for)\b",
            re.I,
        ),
    ),
    (
        "sharing_knowledge",
        re.compile(
            r"\b(did you know|fun fact|here's how|tip:|pro tip:|today i learned|tl;dr)\b",
            re.I,
        ),
    ),
    (
        "reflection",
        re.compile(
            r"\b(i think|i believe|i feel|in my opinion|to me|it seems|from my perspective)\b",
            re.I,
        ),
    ),
    ("other", re.compile(r".*", re.S)),
]

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    log.info("Loading dataset from HuggingFace…")
    ds = load_dataset("filter-with-espresso/moltbook-files", split="train")
    df = ds.to_pandas()
    df["content"] = df["content"].fillna("")
    df["content_len"] = df["content"].str.len()
    df["word_count"] = df["content"].str.split().str.len().fillna(0).astype(int)
    log.info(f"Loaded {len(df):,} posts")
    return df


def parse_comments(comments_str: str) -> list[dict]:
    if not comments_str or comments_str == "[]":
        return []
    try:
        nodes = json.loads(comments_str)
    except json.JSONDecodeError:
        return []
    flat: list[dict] = []

    def _walk(ns, depth=0):
        for n in ns:
            flat.append(
                {
                    "content": n.get("content", ""),
                    "author_name": n.get("author_name"),
                    "lang": n.get("lang"),
                    "depth": depth,
                }
            )
            if n.get("replies"):
                _walk(n["replies"], depth + 1)

    _walk(nodes)
    return flat


# ──────────────────────────────────────────────────────────────────────────────
# A. Language distribution
# ──────────────────────────────────────────────────────────────────────────────


def analyze_language_distribution(df: pd.DataFrame) -> dict:
    log.info("=== A. Language distribution ===")
    lang_counts = df["lang"].value_counts()
    lang_score_by_lang = df.groupby("lang")["lang_score"].mean()

    lang_df = pd.DataFrame(
        {"count": lang_counts, "mean_conf": lang_score_by_lang}
    ).head(20)
    lang_df.to_csv(OUT / "A_language_distribution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    lang_df["count"].head(15).plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].invert_yaxis()
    axes[0].set_title("Top 15 Languages by Post Count")
    axes[0].set_xlabel("Posts")

    lang_df["mean_conf"].head(15).plot.barh(ax=axes[1], color="#55A868")
    axes[1].invert_yaxis()
    axes[1].set_title("Mean Detection Confidence per Language")
    axes[1].set_xlabel("Confidence Score")

    fig.suptitle("Language Distribution (A)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "A_language_distribution.png")
    plt.close(fig)

    # English fraction
    eng_frac = (df["lang"] == "eng_Latn").mean()
    log.info(f"English fraction: {eng_frac:.1%}")
    return {"lang_counts": lang_counts.to_dict(), "english_fraction": eng_frac}


# ──────────────────────────────────────────────────────────────────────────────
# B. Biber-inspired multi-dimensional register analysis
# ──────────────────────────────────────────────────────────────────────────────


def _count_per_100w(text: str, pattern_or_set, mode="regex") -> float:
    words = text.split()
    n = max(len(words), 1)
    if mode == "regex":
        hits = len(pattern_or_set.findall(text))
    else:
        text_l = text.lower()
        hits = sum(1 for w in text_l.split() if w in pattern_or_set)
    return hits / n * 100


def compute_biber_features(text: str) -> dict:
    """Compute a subset of Biber (1988) register features per 100 words."""
    words = text.split()
    n = max(len(words), 1)
    text_l = text.lower()

    # D1 — Informational vs Involved
    nouns_approx = sum(1 for w in words if len(w) > 5 and w[0].isalpha())
    contractions = _count_per_100w(text, CONTRACTIONS)
    first_p = _count_per_100w(text, FIRST_PERSON)
    priv_verbs = sum(1 for w in text_l.split() if w in PRIVATE_VERBS) / n * 100

    # D1 score: high = involved (conversational), low = informational
    d1 = (contractions + first_p + priv_verbs) - (nouns_approx / n * 100)

    # D4 — Persuasion / opinion
    prediction = sum(1 for w in text_l.split() if w in PREDICTION_MODALS) / n * 100
    necessity = sum(1 for w in text_l.split() if w in NECESSITY_MODALS) / n * 100
    second_p = _count_per_100w(text, SECOND_PERSON)
    d4 = prediction + necessity + second_p

    # Hedging density
    hedge_count = sum(1 for tok in text_l.split() if tok in EPISTEMIC_HEDGES)
    hedge_density = hedge_count / n * 100

    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s for s in sentences if len(s.split()) > 2]
    mean_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    type_token = len(set(text_l.split())) / n

    return {
        "d1_involved": d1,
        "d4_persuasion": d4,
        "hedge_density": hedge_density,
        "first_person_rate": first_p,
        "second_person_rate": second_p,
        "contraction_rate": contractions,
        "mean_sent_len": mean_sent_len,
        "ttr": type_token,
        "priv_verb_rate": priv_verbs,
    }


def analyze_biber(df: pd.DataFrame) -> pd.DataFrame:
    log.info("=== B. Biber register analysis ===")
    # English posts only; sample for speed
    eng = df[df["lang"] == "eng_Latn"].copy()
    sample = eng.sample(n=min(30_000, len(eng)), random_state=SEED)

    log.info(f"Computing Biber features on {len(sample):,} posts…")
    features = sample["content"].apply(compute_biber_features).apply(pd.Series)
    features.index = sample.index

    # Add readability
    def fk(t):
        try:
            return textstat.flesch_kincaid_grade(t) if len(t) > 50 else np.nan
        except Exception:
            return np.nan

    features["flesch_kincaid"] = sample["content"].map(fk)
    features["submolt"] = sample["submolt_name"].values

    # Overall statistics
    overall = features.drop(columns="submolt").describe()
    overall.to_csv(OUT / "B_biber_overall.csv")

    # Radar chart of mean Biber features
    dims = [
        "d1_involved",
        "d4_persuasion",
        "hedge_density",
        "first_person_rate",
        "second_person_rate",
        "contraction_rate",
        "mean_sent_len",
        "ttr",
    ]
    means = features[dims].mean()
    stds = features[dims].std()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    means.plot.barh(ax=axes[0], color="#4C72B0", xerr=stds, capsize=3)
    axes[0].set_title("Mean Biber-inspired Features (overall)")
    axes[0].set_xlabel("Value (per 100 words or ratio)")

    # D1 vs D4 scatter by submolt (top 10 submolts)
    top10 = df["submolt_name"].value_counts().head(10).index.tolist()
    sub_means = (
        features[features["submolt"].isin(top10)]
        .groupby("submolt")[["d1_involved", "d4_persuasion", "hedge_density"]]
        .mean()
    )
    scatter = axes[1].scatter(
        sub_means["d1_involved"],
        sub_means["d4_persuasion"],
        s=sub_means["hedge_density"] * 500,
        c=range(len(sub_means)),
        cmap="tab10",
        alpha=0.8,
    )
    for sub, row in sub_means.iterrows():
        axes[1].annotate(
            sub,
            (row["d1_involved"], row["d4_persuasion"]),
            fontsize=7,
            ha="center",
            va="bottom",
        )
    axes[1].set_xlabel("D1 Involved score (higher = conversational)")
    axes[1].set_ylabel("D4 Persuasion score")
    axes[1].set_title("Register Space per Submolt\n(bubble = hedging density)")

    fig.suptitle("Biber Register Analysis (B)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "B_biber_register.png")
    plt.close(fig)

    sub_means.to_csv(OUT / "B_biber_by_submolt.csv")
    log.info(f"Biber overall means:\n{means.round(3).to_string()}")
    return features


# ──────────────────────────────────────────────────────────────────────────────
# C. AI-agent jargon & persona construction
# ──────────────────────────────────────────────────────────────────────────────


def analyze_agent_jargon(df: pd.DataFrame) -> None:
    log.info("=== C. Agent jargon & persona ===")
    eng = df[df["lang"] == "eng_Latn"].copy()
    texts_l = eng["content"].str.lower()

    jargon_counts = {}
    for term in AGENT_JARGON:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b")
        jargon_counts[term] = texts_l.str.count(pattern).sum()

    jargon_series = pd.Series(jargon_counts).sort_values(ascending=False)
    jargon_series.to_csv(OUT / "C_jargon_counts.csv", header=["count"])

    # Persona-construction phrases (regex extractions)
    intro_pattern = re.compile(
        r"(i'?m|i am|my name is)\s+([A-Z][A-Za-z0-9_\-]+).*?"
        r"(ai|agent|assistant|bot|running on|openclaw|clawdbot|claude|gpt)",
        re.I | re.S,
    )
    claimed_pattern = re.compile(r"my human\s+([A-Z][A-Za-z]+)", re.I)
    platform_pattern = re.compile(r"running on\s+([A-Za-z0-9_\-]+)", re.I)

    intro_matches = texts_l.str.count(
        re.compile(r"\bi am\b|\bi'?m\b.*\bai\b|\bagent\b", re.I)
    )
    claimed_matches = texts_l.str.count(re.compile(r"\bmy human\b", re.I))
    platform_all = []
    for text in eng["content"]:
        platform_all.extend(platform_pattern.findall(text))
    platform_counts = Counter(p.lower() for p in platform_all)

    # Posts with self-introductions
    intro_posts = int((intro_matches > 0).sum())
    claimed_posts = int((claimed_matches > 0).sum())

    log.info(
        f"Posts with 'I am/I'm AI/agent': {intro_posts:,} ({intro_posts / len(eng):.1%})"
    )
    log.info(
        f"Posts with 'my human': {claimed_posts:,} ({claimed_posts / len(eng):.1%})"
    )
    log.info(f"Top platforms: {platform_counts.most_common(10)}")

    pd.DataFrame(
        platform_counts.most_common(20), columns=["platform", "mentions"]
    ).to_csv(OUT / "C_platform_mentions.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    jargon_series.head(25).plot.barh(ax=axes[0], color="#C44E52")
    axes[0].invert_yaxis()
    axes[0].set_title("Top 25 AI-Agent Jargon Terms (English posts)")
    axes[0].set_xlabel("Occurrences")

    plat_df = pd.DataFrame(
        platform_counts.most_common(15), columns=["platform", "count"]
    )
    if not plat_df.empty:
        plat_df.set_index("platform")["count"].plot.barh(ax=axes[1], color="#4C72B0")
        axes[1].invert_yaxis()
        axes[1].set_title("Agent Platforms ('running on …')")

    fig.suptitle("AI-Agent Jargon & Persona (C)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "C_agent_jargon.png")
    plt.close(fig)

    return {
        "jargon_counts": jargon_series.to_dict(),
        "intro_posts_pct": intro_posts / len(eng),
        "claimed_posts_pct": claimed_posts / len(eng),
        "top_platforms": dict(platform_counts.most_common(10)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# D. Speech-act classification
# ──────────────────────────────────────────────────────────────────────────────


def classify_speech_act(text: str) -> str:
    t = text.strip()
    for label, pattern in SPEECH_ACTS:
        if pattern.search(t):
            return label
    return "other"


def analyze_speech_acts(df: pd.DataFrame) -> None:
    log.info("=== D. Speech-act classification ===")
    eng = df[df["lang"] == "eng_Latn"].copy()

    eng["speech_act"] = eng["content"].map(classify_speech_act)
    act_counts = eng["speech_act"].value_counts()
    act_counts.to_csv(OUT / "D_speech_acts.csv", header=["count"])

    # By submolt
    top10 = df["submolt_name"].value_counts().head(10).index
    sub_acts = (
        eng[eng["submolt_name"].isin(top10)]
        .groupby(["submolt_name", "speech_act"])
        .size()
        .unstack(fill_value=0)
    )
    sub_acts_norm = sub_acts.div(sub_acts.sum(axis=1), axis=0)
    sub_acts_norm.to_csv(OUT / "D_speech_acts_by_submolt.csv")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    act_counts.plot.barh(ax=axes[0], color=sns.color_palette("husl", len(act_counts)))
    axes[0].invert_yaxis()
    axes[0].set_title("Speech Act Distribution (English posts)")

    if not sub_acts_norm.empty:
        sub_acts_norm.plot.barh(stacked=True, ax=axes[1], colormap="tab10", width=0.8)
        axes[1].legend(fontsize=7, loc="lower right")
        axes[1].set_title("Speech Acts by Top Submolt (normalised)")

    fig.suptitle("Speech Act Analysis (D)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "D_speech_acts.png")
    plt.close(fig)
    log.info(f"Speech act distribution:\n{act_counts.to_string()}")


# ──────────────────────────────────────────────────────────────────────────────
# E. POS profile & syntactic complexity (spaCy)
# ──────────────────────────────────────────────────────────────────────────────


def analyze_pos_spacy(df: pd.DataFrame) -> pd.DataFrame:
    log.info("=== E. POS & syntactic complexity (spaCy) ===")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.max_length = 2_000_000

    eng = df[df["lang"] == "eng_Latn"].copy()
    # Sample — spaCy parse is expensive
    sample = eng.sample(n=min(10_000, len(eng)), random_state=SEED)
    texts = sample["content"].str[:512].tolist()

    rows = []
    log.info(f"Parsing {len(texts):,} texts with spaCy…")
    for doc in nlp.pipe(texts, batch_size=128, n_process=1):
        tokens = [t for t in doc if not t.is_space]
        n = max(len(tokens), 1)
        pos_counts = Counter(t.pos_ for t in tokens)

        # Dependency depth (max depth in sentence)
        depths = []
        for sent in doc.sents:

            def _depth(tok, d=0):
                if list(tok.children):
                    return max(_depth(c, d + 1) for c in tok.children)
                return d

            root = [t for t in sent if t.dep_ == "ROOT"]
            if root:
                depths.append(_depth(root[0]))

        rows.append(
            {
                "noun_rate": pos_counts.get("NOUN", 0) / n,
                "verb_rate": pos_counts.get("VERB", 0) / n,
                "adj_rate": pos_counts.get("ADJ", 0) / n,
                "adv_rate": pos_counts.get("ADV", 0) / n,
                "propn_rate": pos_counts.get("PROPN", 0) / n,
                "punct_rate": pos_counts.get("PUNCT", 0) / n,
                "num_rate": pos_counts.get("NUM", 0) / n,
                "token_count": n,
                "mean_dep_depth": np.mean(depths) if depths else 0,
                "max_dep_depth": max(depths) if depths else 0,
                "n_sents": len(list(doc.sents)),
                "submolt": sample["submolt_name"].iloc[len(rows)],
            }
        )

    pos_df = pd.DataFrame(rows)
    overall = pos_df.drop(columns="submolt").describe()
    overall.to_csv(OUT / "E_pos_overall.csv")

    top10 = df["submolt_name"].value_counts().head(10).index
    sub_pos = (
        pos_df[pos_df["submolt"].isin(top10)]
        .groupby("submolt")[
            [
                "noun_rate",
                "verb_rate",
                "adj_rate",
                "adv_rate",
                "propn_rate",
                "mean_dep_depth",
            ]
        ]
        .mean()
    )
    sub_pos.to_csv(OUT / "E_pos_by_submolt.csv")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # POS rates overall (mean)
    pos_means = pos_df[
        ["noun_rate", "verb_rate", "adj_rate", "adv_rate", "propn_rate", "num_rate"]
    ].mean()
    pos_means.plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].set_title("Mean POS Rates (English sample)")
    axes[0].invert_yaxis()

    # Dependency depth distribution
    pos_df["mean_dep_depth"].clip(upper=10).hist(
        bins=30, ax=axes[1], color="#55A868", edgecolor="white"
    )
    axes[1].set_title("Mean Dependency Depth per Post")
    axes[1].set_xlabel("Depth")

    # Submolt POS heatmap
    if not sub_pos.empty:
        sns.heatmap(
            sub_pos,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=axes[2],
            cbar_kws={"shrink": 0.8},
        )
        axes[2].set_title("POS Profile by Submolt")

    fig.suptitle(
        "POS Profile & Syntactic Complexity (E)", fontsize=13, fontweight="bold"
    )
    fig.savefig(OUT / "E_pos_profile.png")
    plt.close(fig)
    log.info(f"POS means:\n{pos_means.round(3).to_string()}")
    return pos_df


# ──────────────────────────────────────────────────────────────────────────────
# F. Named-entity landscape
# ──────────────────────────────────────────────────────────────────────────────


def analyze_ner(df: pd.DataFrame) -> None:
    log.info("=== F. Named-entity landscape ===")
    nlp = spacy.load("en_core_web_sm")

    eng = df[df["lang"] == "eng_Latn"].copy()
    sample = eng.sample(n=min(8_000, len(eng)), random_state=SEED)
    texts = sample["content"].str[:512].tolist()

    entity_records = []
    log.info(f"Running NER on {len(texts):,} texts…")
    for doc in nlp.pipe(texts, batch_size=64, n_process=1):
        for ent in doc.ents:
            entity_records.append({"text": ent.text, "label": ent.label_})

    if not entity_records:
        log.warning("No entities found")
        return

    ent_df = pd.DataFrame(entity_records)
    ent_type_counts = ent_df["label"].value_counts()

    # Top entities per type
    top_per_type = {}
    for label, group in ent_df.groupby("label"):
        top = group["text"].str.lower().value_counts().head(10)
        top_per_type[label] = top.to_dict()

    pd.DataFrame({lab: pd.Series(vals) for lab, vals in top_per_type.items()}).to_csv(
        OUT / "F_top_entities_per_type.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ent_type_counts.plot.barh(ax=axes[0], color="#C44E52")
    axes[0].invert_yaxis()
    axes[0].set_title("Entity Type Distribution")

    # Top ORG and PRODUCT entities
    for lab in ["ORG", "PRODUCT", "GPE", "PERSON"]:
        if lab in top_per_type:
            vals = pd.Series(top_per_type[lab]).head(15)
            if not vals.empty:
                break
    vals.plot.barh(ax=axes[1], color="#55A868")
    axes[1].invert_yaxis()
    axes[1].set_title(f"Top '{lab}' Entities")

    fig.suptitle("Named-Entity Landscape (F)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "F_named_entities.png")
    plt.close(fig)
    log.info(f"Entity types:\n{ent_type_counts.to_string()}")


# ──────────────────────────────────────────────────────────────────────────────
# G. Register variation across top communities
# ──────────────────────────────────────────────────────────────────────────────


def analyze_community_register(df: pd.DataFrame, biber_df: pd.DataFrame) -> None:
    log.info("=== G. Community register variation ===")
    top15 = df["submolt_name"].value_counts().head(15).index

    metrics = [
        "d1_involved",
        "d4_persuasion",
        "hedge_density",
        "flesch_kincaid",
        "ttr",
        "mean_sent_len",
        "first_person_rate",
    ]

    sub_stats = (
        biber_df[biber_df["submolt"].isin(top15)]
        .groupby("submolt")[metrics]
        .agg(["mean", "std"])
    )
    sub_stats.columns = ["_".join(c) for c in sub_stats.columns]
    sub_stats.to_csv(OUT / "G_community_register.csv")

    # Heatmap of z-scored means
    means = biber_df[biber_df["submolt"].isin(top15)].groupby("submolt")[metrics].mean()
    scaler = StandardScaler()
    z_means = pd.DataFrame(
        scaler.fit_transform(means),
        index=means.index,
        columns=means.columns,
    )

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        z_means,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    ax.set_title(
        "Register Profile by Community (z-scored, top 15 submolts)\n"
        "Red = higher than average | Blue = lower",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(OUT / "G_community_register_heatmap.png")
    plt.close(fig)
    log.info("Community register heatmap saved")


# ──────────────────────────────────────────────────────────────────────────────
# H. Hedging & epistemic language
# ──────────────────────────────────────────────────────────────────────────────


def analyze_hedging(df: pd.DataFrame) -> None:
    log.info("=== H. Hedging & epistemic language ===")
    eng = df[df["lang"] == "eng_Latn"].copy()

    hedge_pattern = re.compile(
        r"\b("
        + "|".join(
            re.escape(h) for h in sorted(EPISTEMIC_HEDGES, key=len, reverse=True)
        )
        + r")\b",
        re.I,
    )

    eng["hedge_count"] = eng["content"].str.count(hedge_pattern)
    eng["hedge_rate"] = eng["hedge_count"] / eng["word_count"].clip(lower=1) * 100

    per_hedge = Counter()
    for text in eng["content"]:
        for m in hedge_pattern.finditer(text.lower()):
            per_hedge[m.group()] += 1

    hedge_freq = pd.Series(per_hedge).sort_values(ascending=False)
    hedge_freq.to_csv(OUT / "H_hedge_frequency.csv", header=["count"])

    # By submolt
    top10 = df["submolt_name"].value_counts().head(10).index
    sub_hedge = (
        eng[eng["submolt_name"].isin(top10)]
        .groupby("submolt_name")["hedge_rate"]
        .agg(["mean", "median", "std"])
    )
    sub_hedge.to_csv(OUT / "H_hedge_by_submolt.csv")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    hedge_freq.head(25).plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].invert_yaxis()
    axes[0].set_title("Top 25 Hedge Tokens (English posts)")
    axes[0].set_xlabel("Occurrences")

    eng["hedge_rate"].clip(upper=10).hist(
        bins=50, ax=axes[1], color="#55A868", edgecolor="white"
    )
    axes[1].set_title("Hedging Rate Distribution")
    axes[1].set_xlabel("Hedge tokens per 100 words")

    if not sub_hedge.empty:
        sub_hedge["mean"].sort_values().plot.barh(ax=axes[2], color="#C44E52")
        axes[2].set_title("Mean Hedging Rate by Submolt")
        axes[2].set_xlabel("Hedge tokens per 100 words")

    fig.suptitle("Hedging & Epistemic Language (H)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "H_hedging.png")
    plt.close(fig)
    log.info(f"Overall hedge rate: {eng['hedge_rate'].mean():.2f} per 100 words")


# ──────────────────────────────────────────────────────────────────────────────
# I. Conversational vs broadcast posts
# ──────────────────────────────────────────────────────────────────────────────


def analyze_conversational_vs_broadcast(df: pd.DataFrame) -> None:
    log.info("=== I. Conversational vs broadcast ===")
    eng = df[df["lang"] == "eng_Latn"].copy()

    # Broadcast markers
    eng["is_broadcast"] = eng["content"].str.contains(
        r"\b(announcing|just released|new:|update:|breaking:|alert:)\b",
        case=False,
    ) | eng["content"].str.contains(r"https?://", case=False)
    # Conversational markers
    eng["is_conversational"] = (
        eng["content"].str.contains(r"\?")
        | eng["content"].str.contains(r"^(hey|hi|hello)\b", case=False)
        | eng["content"].str.contains(r"\b(what do you|have you|do you)\b", case=False)
    )
    eng["type"] = "neutral"
    eng.loc[eng["is_broadcast"] & ~eng["is_conversational"], "type"] = "broadcast"
    eng.loc[eng["is_conversational"] & ~eng["is_broadcast"], "type"] = "conversational"
    eng.loc[eng["is_broadcast"] & eng["is_conversational"], "type"] = "hybrid"

    type_counts = eng["type"].value_counts()
    type_counts.to_csv(OUT / "I_post_types.csv", header=["count"])

    # Compare upvotes, comment count by type
    type_stats = eng.groupby("type")[["upvotes", "comment_count", "word_count"]].mean()
    type_stats.to_csv(OUT / "I_type_stats.csv")

    # By submolt
    top10 = df["submolt_name"].value_counts().head(10).index
    sub_type = (
        eng[eng["submolt_name"].isin(top10)]
        .groupby(["submolt_name", "type"])
        .size()
        .unstack(fill_value=0)
    )
    sub_type_norm = sub_type.div(sub_type.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    type_counts.plot.bar(ax=axes[0], color=["#4C72B0", "#55A868", "#C44E52", "#dd8452"])
    axes[0].set_title("Post Type Distribution")
    axes[0].tick_params(axis="x", rotation=30)

    type_stats[["upvotes", "comment_count"]].plot.bar(ax=axes[1])
    axes[1].set_title("Mean Upvotes & Comments by Post Type")
    axes[1].tick_params(axis="x", rotation=30)

    if not sub_type_norm.empty:
        sub_type_norm.plot.barh(stacked=True, ax=axes[2], colormap="tab10")
        axes[2].set_title("Post Type Mix by Submolt")
        axes[2].legend(fontsize=7)

    fig.suptitle("Conversational vs Broadcast (I)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "I_conversational_vs_broadcast.png")
    plt.close(fig)
    log.info(f"Post type distribution:\n{type_counts.to_string()}")
    return type_stats


# ──────────────────────────────────────────────────────────────────────────────
# J. Summary report
# ──────────────────────────────────────────────────────────────────────────────


def write_report(
    lang_stats: dict,
    biber_df: pd.DataFrame,
    jargon_stats: dict,
    type_stats: pd.DataFrame,
) -> None:
    log.info("=== J. Writing summary report ===")

    eng_biber = biber_df.drop(columns="submolt").describe()

    report_lines = [
        "# SOTA Linguistic Analysis — Moltbook AI-Agent Social Network",
        "",
        f"**Dataset**: {lang_stats['lang_counts'].get('eng_Latn', 0):,} English posts "
        f"({lang_stats['english_fraction']:.1%} of corpus)",
        "",
        "## A. Language Distribution",
        f"- Dominant language: `eng_Latn` ({lang_stats['english_fraction']:.1%})",
        "- See A_language_distribution.csv for full breakdown.",
        "",
        "## B. Biber Register Dimensions",
        f"- **D1 Involved score** (mean): {eng_biber.loc['mean', 'd1_involved']:.3f}  ",
        "  (positive = conversational/involved; negative = informational/dense)",
        f"- **D4 Persuasion score** (mean): {eng_biber.loc['mean', 'd4_persuasion']:.3f}",
        f"- **Hedge density** (mean per 100w): {eng_biber.loc['mean', 'hedge_density']:.3f}",
        f"- **Mean sentence length**: {eng_biber.loc['mean', 'mean_sent_len']:.1f} words",
        f"- **Type-token ratio**: {eng_biber.loc['mean', 'ttr']:.3f}",
        f"- **Mean Flesch-Kincaid grade**: {eng_biber.loc['mean', 'flesch_kincaid']:.1f}",
        "",
        "## C. AI-Agent Jargon & Persona",
        f"- Posts with self-identification ('I am/I'm … AI/agent'): "
        f"{jargon_stats['intro_posts_pct']:.1%}",
        f"- Posts mentioning 'my human': {jargon_stats['claimed_posts_pct']:.1%}",
        f"- Top jargon terms: {list(jargon_stats['jargon_counts'].keys())[:10]}",
        f"- Top platforms: {list(jargon_stats['top_platforms'].keys())[:5]}",
        "",
        "## D. Speech Acts",
        "  See D_speech_acts.csv and D_speech_acts.png",
        "",
        "## E. POS Profile",
        "  See E_pos_overall.csv and E_pos_profile.png",
        "",
        "## F. Named Entities",
        "  See F_top_entities_per_type.csv and F_named_entities.png",
        "",
        "## G. Community Register Variation",
        "  See G_community_register_heatmap.png",
        "",
        "## H. Hedging Language",
        "  See H_hedging.png and H_hedge_frequency.csv",
        "",
        "## I. Conversational vs Broadcast",
    ]
    if type_stats is not None:
        for t, row in type_stats.iterrows():
            report_lines.append(
                f"  - **{t}**: mean upvotes={row['upvotes']:.2f}, "
                f"mean comments={row['comment_count']:.2f}, "
                f"mean words={row['word_count']:.0f}"
            )
    report_lines += [
        "",
        "---",
        "_Generated by language_style_analysis.py_",
    ]

    report_text = "\n".join(report_lines)
    (OUT / "J_summary_report.md").write_text(report_text)
    log.info(f"Report written to {OUT / 'J_summary_report.md'}")
    print("\n" + "=" * 70)
    print(report_text)
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    df = load_data()

    lang_stats = analyze_language_distribution(df)
    biber_df = analyze_biber(df)
    jargon_stats = analyze_agent_jargon(df)
    analyze_speech_acts(df)
    pos_df = analyze_pos_spacy(df)
    analyze_ner(df)
    analyze_community_register(df, biber_df)
    analyze_hedging(df)
    type_stats = analyze_conversational_vs_broadcast(df)
    write_report(lang_stats, biber_df, jargon_stats, type_stats)

    log.info(f"All outputs written to {OUT}/")


if __name__ == "__main__":
    main()
