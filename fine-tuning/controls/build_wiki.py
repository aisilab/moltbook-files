"""Rebuild the Wikipedia control sample and check it against the original run's fingerprints.

Requires datasets==4.3.0 (pinned by Unsloth 2026.9.x): the streaming shuffle changed in 5.x,
so the same seed gives other articles there.

Usage:  python build_wiki.py [out.parquet]
"""

import sys

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

OUT = sys.argv[1] if len(sys.argv) > 1 else "wikipedia_control.parquet"
EXPECT = {"rows": 232_498, "mean_chars": 2327, "tokens": 113_356_773,
          "first_titles": ["Macrocoma", "1900 United States presidential election in Vermont"]}

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
df = pd.DataFrame([{"title": e["title"], "content": e["text"]} for e in ds.shuffle(seed=0).take(232_498)])
df.to_parquet(OUT, index=False)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
texts = [f"{t.strip()}\n\n{c.strip()}" if t.strip() and c.strip() else (t.strip() or c.strip())
         for t, c in zip(df.title.astype(str), df.content.astype(str))]
tokens = sum(min(len(x), 2048) for i in range(0, len(texts), 2000)
             for x in tok(texts[i:i + 2000], add_special_tokens=False)["input_ids"])
got = {"rows": len(df), "mean_chars": round((df.title.str.len() + df.content.str.len() + 2).mean()),
       "tokens": tokens, "first_titles": list(df.title[:2])}
for k in EXPECT:
    print(f"{'OK  ' if got[k] == EXPECT[k] else 'FAIL'} {k:<13} got={got[k]}  expected={EXPECT[k]}")
print("IDENTICAL SAMPLE" if got == EXPECT else "SAMPLE DIFFERS")
