# moltbook-scraper

Reference implementation that produces the **Moltbook Files** dataset — 232,497 social-media posts and 2.2 M nested comments scraped from [moltbook.com](https://www.moltbook.com), language-tagged, and PII-anonymized.

| | |
|---|---|
| Dataset | https://huggingface.co/datasets/aisilab/moltbook-files |
| Datasheet | [`DATASHEET.md`](DATASHEET.md) |
| Code license | MIT (see [`LICENSE`](LICENSE)) |
| Data license | ODC-BY 1.0 |

## What's in the dataset

| | |
|---|---|
| Posts | 232,497 |
| Comments (incl. replies) | 2,202,950 |
| Communities (submolts) | 3,628 |
| Unique authors | 34,905 |
| Date range of `created_at` | 2026-01-27 — 2026-02-07 |
| Languages | tagged with fastText `lid.176`; multilingual |

Schema, retained fields, and ethical considerations are documented in [`DATASHEET.md`](DATASHEET.md).

## Pipeline

```
crawl  ──▶  scrape  ──▶  process  ──▶  push  ──▶  analyze
 │            │            │            │           │
 IDs         raw JSON    cleaned       HF parquet  plots + stats
 (queue)    (per post)   + anon.       upload      (data/plots/)
```

| Stage | Script | Output |
|---|---|---|
| 1. Crawl | [`scripts/crawl.py`](scripts/crawl.py) | `data/metadata/posts_queue.jsonl` |
| 2. Scrape | [`scripts/scrape.py`](scripts/scrape.py) | `data/raw/{uuid}.json` (~2.5 GB) |
| 3. Process | [`scripts/process.py`](scripts/process.py) | `data/processed/{uuid}.json`, `data/metadata/processing_stats.json` |
| 4. Push | [`scripts/push_dataset.py`](scripts/push_dataset.py) | `aisilab/moltbook-files` on HuggingFace |
| 5. Analyze | [`scripts/analysis.py`](scripts/analysis.py) | `data/plots/*.png`, `data/plots/stats.csv` |

Stages 1–3 are independent and idempotent — each will skip work that has already been completed (queue dedup, file existence). Re-running step 4 overwrites the published snapshot.

`scripts/extras/` contains optional analyses that consume the published HF dataset:
- [`scripts/extras/lit_check.py`](scripts/extras/lit_check.py) — exports the Lithuanian (`lit_Latn`) subset.
- [`scripts/extras/new_langs.py`](scripts/extras/new_langs.py) — LLM-based new-language detection. Requires an OpenAI-compatible endpoint (`API_KEY` + `API_BASE`). The output is published separately at `aisilab/moltbook-files-new-language-signals`.

## Reproduction

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)
- ~5 GB free disk for raw + processed JSON
- An HF account + token with write access if you want to re-push the dataset

### Setup

```bash
git clone https://github.com/williambrach/moltbook-scraper && cd moltbook-scraper
uv sync --group analysis
cp .env.example .env
# edit .env: set HF_TOKEN
```


## Ethics, ToS, and rate limiting

- **Source.** All data comes from Moltbook's public REST API (`/api/v1/posts`), the same endpoint that serves Moltbook's own web client. We do not bypass authentication, scrape private submolts, or access non-public profile information.
- **Rate limits.** [`scripts/crawl.py`](scripts/crawl.py) sleeps 1 s between requests (`SLEEP_SECONDS = 1.0`); [`scripts/scrape.py`](scripts/scrape.py) sends 64 concurrent requests then sleeps 1 s before the next batch. We did not observe rate-limiting from the platform.
- **PII.** Free-text content (post title, post body, every comment body) is passed through [Microsoft Presidio](https://github.com/microsoft/presidio) with custom recognizers for emails, phone numbers, crypto addresses, US SSNs / ITINs, API keys, passwords, credit cards, IBANs, and BIP-39 seed phrases. See [`scripts/process.py`](scripts/process.py) and the [datasheet](DATASHEET.md#preprocessing--cleaning--labelling) for counts.
- **Usernames.** `author_name` and `comment.author_name` are retained as the public-display handles already exposed by Moltbook's API. They are *not* anonymized. If you are a Moltbook user and want your data removed from the next snapshot, email <wibrach@gmail.com> with your `author_name` or `post_id`.
- **Spam / blocklist.** Posts that match the heuristic spam detector or the blocklist in `data/metadata/filter_words.txt` are kept as rows but their textual fields are replaced with `<REMOVED-SPAM>` / `<REMOVED-BLOCKLIST>` sentinels.

## Layout

```
moltbook-scraper/
├── scripts/
│   ├── crawl.py            # discover post IDs
│   ├── scrape.py           # fetch full posts
│   ├── process.py          # clean + anonymize
│   ├── push_dataset.py     # publish to HF
│   ├── analysis.py         # plots + stats
│   └── extras/             # optional analyses (LLM-based)
│       ├── lit_check.py
│       └── new_langs.py
├── data/
│   ├── raw/                # post JSONs (gitignored)
│   ├── processed/          # cleaned JSONs (gitignored)
│   ├── metadata/           # queues, stats, models (gitignored)
│   └── plots/              # analysis output (committed)
├── DATASHEET.md
├── CITATION.cff
├── LICENSE                 # MIT (code)
├── Makefile
├── pyproject.toml
└── README.md
```

## Contact

William Brach &mdash; <wiliam.brach@stuba.sk>
