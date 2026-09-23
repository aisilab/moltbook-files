"""Count the training tokens of each dataset, as reported in the training-tokens table.

Serializes every example the way train_control.py does (title, blank line, body), tokenizes it with
the Qwen2.5 tokenizer and truncates at 2,048 tokens, the training sequence length.

Usage:  python count_tokens.py wikipedia_control.parquet
"""

import sys

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

MAX_SEQ_LEN = 2048
N_EXAMPLES = 232_498  # Moltbook post count; the Reddit and Wikipedia controls use the same number
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")


def serialize(df: pd.DataFrame) -> list[str]:
    texts = []
    for title, content in zip(df["title"].astype(str).str.strip(), df["content"].astype(str).str.strip()):
        if title or content:
            texts.append(f"{title}\n\n{content}" if title and content else title or content)
    return texts


def report(name: str, texts: list[str]) -> None:
    raw = trained = 0
    for i in range(0, len(texts), 2000):
        for ids in tok(texts[i:i + 2000], add_special_tokens=False)["input_ids"]:
            raw += len(ids)
            trained += min(len(ids), MAX_SEQ_LEN)
    print(f"{name:<10} examples={len(texts):>8,}  tokens={trained:>12,}  mean={trained / len(texts):6.1f}"
          f"  truncated={100 * (raw - trained) / raw:5.2f}%")


moltbook = load_dataset("aisilab/moltbook-files", split="train").to_pandas()
reddit = load_dataset("tensorshield/reddit_dataset_157", split="train", streaming=True)
report("Moltbook", serialize(moltbook))
report("Reddit", [e["text"] for e in reddit.shuffle(seed=0).take(N_EXAMPLES)])
report("Wikipedia", serialize(pd.read_parquet(sys.argv[1])))
