"""Build the non-social-media control set for the ARR rebuttal.

Samples Wikipedia articles into a parquet with the same `title`/`content` schema as the
Moltbook posts, so `run_finetune.load_parquet_as_text` serializes it through the same code
path (`title\\n\\ncontent`, plain text, no chat template).

Also reports token totals per dataset, which reviewer apcz asked for directly.

Usage:
    python prepare_wikipedia_control.py --out data/wikipedia_control.parquet
    python prepare_wikipedia_control.py --count-tokens-only \\
        --moltbook data/moltbook_files/train-00000-of-00001.parquet
"""

import argparse

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

WIKI = ("wikimedia/wikipedia", "20231101.en")
N_SAMPLES = 232_498  # Moltbook post count, as used for the Reddit control
TOKENIZER = "Qwen/Qwen2.5-14B-Instruct"
MAX_SEQ_LEN = 2048  # training truncation; tokens past this are never seen


def build(n_samples: int, seed: int, out: str) -> pd.DataFrame:
    ds = load_dataset(*WIKI, split="train", streaming=True).shuffle(seed=seed).take(n_samples)
    rows = [{"title": e["title"], "content": e["text"]} for e in ds]
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    return df


def serialize(df: pd.DataFrame) -> list[str]:
    """Mirror load_parquet_as_text so the token count reflects what training actually sees."""
    texts = []
    for _, row in df.iterrows():
        title, content = str(row.get("title", "")).strip(), str(row.get("content", "")).strip()
        if title and content:
            texts.append(f"{title}\n\n{content}")
        elif title or content:
            texts.append(title or content)
    return texts


def count_tokens(texts: list[str], label: str, tok) -> None:
    raw = trained = 0
    for i in range(0, len(texts), 1000):
        for ids in tok(texts[i : i + 1000], add_special_tokens=False)["input_ids"]:
            raw += len(ids)
            trained += min(len(ids), MAX_SEQ_LEN)
    n = len(texts)
    print(
        f"{label:<12} n={n:>8,}  raw={raw:>14,}  after truncation={trained:>14,}  "
        f"mean={trained / n:>7.1f}  truncated={100 * (raw - trained) / raw:>5.2f}%"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/wikipedia_control.parquet")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--moltbook", help="Moltbook parquet, for the token comparison")
    p.add_argument("--reddit-samples", type=int, default=N_SAMPLES)
    p.add_argument("--count-tokens-only", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    if not args.count_tokens_only:
        wiki = build(args.n_samples, args.seed, args.out)
    else:
        wiki = pd.read_parquet(args.out)

    print(f"\nTraining tokens per condition (Qwen2.5-14B tokenizer, cap {MAX_SEQ_LEN}/example):")
    if args.moltbook:
        count_tokens(serialize(pd.read_parquet(args.moltbook)), "Moltbook", tok)

    reddit = load_dataset("tensorshield/reddit_dataset_157", split="train", streaming=True)
    reddit = reddit.shuffle(seed=args.seed).take(args.reddit_samples)
    count_tokens([e["text"] for e in reddit], "Reddit", tok)

    count_tokens(serialize(wiki), "Wikipedia", tok)


if __name__ == "__main__":
    main()
