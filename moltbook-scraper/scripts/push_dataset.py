import io
import os
from datetime import datetime
from pathlib import Path

import orjson
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi
from tqdm import tqdm

load_dotenv()

REPO_ID = "aisilab/moltbook-files"
PROCESSED_DIR = Path("data/processed")
STATS_PATH = Path("data/metadata/processing_stats.json")

COMMENT_KEEP_KEYS = {
    "id",
    "content",
    "parent_id",
    "upvotes",
    "downvotes",
    "created_at",
    "author_id",
    "replies",
    "lang",
    "lang_score",
}


def clean_comment(comment: dict) -> dict:
    """Strip author object from a comment. Recurse into replies."""
    cleaned = {k: v for k, v in comment.items() if k in COMMENT_KEEP_KEYS}
    author = comment.get("author") or {}
    cleaned["author_name"] = author.get("name")
    cleaned["replies"] = [clean_comment(r) for r in comment.get("replies", [])]
    return cleaned


def flatten_post(data: dict) -> dict | None:
    """Convert a processed JSON file into a flat row dict."""
    post = data.get("post")
    if not post:
        return None

    comments = data.get("comments", [])
    cleaned_comments = [clean_comment(c) for c in comments]

    submolt = post.get("submolt") or {}
    author = post.get("author") or {}

    return {
        "post_id": post.get("id"),
        "title": post.get("title"),
        "content": post.get("content"),
        "url": post.get("url"),
        "upvotes": post.get("upvotes"),
        "downvotes": post.get("downvotes"),
        "comment_count": post.get("comment_count"),
        "created_at": post.get("created_at"),
        "submolt_id": submolt.get("id"),
        "submolt_name": submolt.get("name"),
        "author_id": author.get("id"),
        "author_name": author.get("name"),
        "lang": post.get("lang"),
        "lang_score": post.get("lang_score"),
        "comments": orjson.dumps(cleaned_comments).decode(),
    }


REPO_URL = "https://github.com/williambrach/moltbook-scraper"
MAINTAINER_EMAIL = "wibrach@gmail.com"


def build_readme(stats: dict | None, num_rows: int) -> str:
    snapshot_date = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [
        "---",
        "license: odc-by",
        "task_categories:",
        "  - text-classification",
        "language:",
        "  - en",
        "size_categories:",
        "  - 100K<n<1M",
        "---",
        "",
        "# Moltbook Files",
        "",
        f"Social-media posts and nested comment threads from [moltbook.com](https://www.moltbook.com), produced by [{REPO_URL}]({REPO_URL}). **{num_rows:,}** posts in this snapshot ({snapshot_date}).",
        "",
        f"For full provenance, ethical considerations, and limitations see the [datasheet]({REPO_URL}/blob/main/DATASHEET.md).",
        "",
        "## Schema",
        "",
        "| Column | Type | Notes |",
        "|---|---|---|",
        "| `post_id` | string | UUID |",
        "| `title` | string | PII-anonymized |",
        "| `content` | string | PII-anonymized; spam/blocklist replaced with sentinels |",
        "| `url` | string | original Moltbook URL |",
        "| `upvotes` | int | |",
        "| `downvotes` | int | |",
        "| `comment_count` | int | |",
        "| `created_at` | string | ISO 8601 |",
        "| `submolt_id` | string | community UUID |",
        "| `submolt_name` | string | community handle |",
        "| `author_id` | string | author UUID |",
        "| `author_name` | string | public-display username (see Privacy) |",
        "| `lang` | string | fastText language code |",
        "| `lang_score` | float | fastText confidence |",
        "| `comments` | string (JSON) | nested comment tree (see datasheet) |",
        "",
        "## Privacy & retained fields",
        "",
        "Free-text fields (`title`, `content`, comment bodies) are passed through Microsoft Presidio with custom recognizers for emails, phone numbers, crypto addresses, US SSNs/ITINs, API keys (`sk-`), passwords (`password:` patterns), credit cards, IBANs, and BIP-39 seed phrases. Detected spans are replaced with `<ENTITY_TYPE>` placeholders. See `scripts/process.py` in the source repository.",
        "",
        f"`author_name` and `comment.author_name` are retained because they are public-display handles already exposed by Moltbook's API. To request removal of your data, email <{MAINTAINER_EMAIL}> with the affected `post_id` or `author_name` and we will exclude the matching rows from the next snapshot.",
        "",
        "## Source data terms",
        "",
        "Posts and comments are public-API content from [moltbook.com](https://www.moltbook.com). Users of this dataset must additionally comply with Moltbook's Terms of Service when re-using the underlying content.",
        "",
    ]

    if stats:
        lines.extend([
            "## Processing Stats",
            "",
            f"- **Total files processed**: {stats.get('total_files', 'N/A'):,}",
            f"- **Total content fields**: {stats.get('total_fields', 'N/A'):,}",
            f"- **Normalized (changed)**: {stats.get('normalized_changed', 'N/A'):,}",
            f"- **Removed (spam)**: {stats.get('removed_spam', 'N/A'):,}",
            f"- **Removed (blocklist)**: {stats.get('removed_blocklist', 'N/A'):,}",
            f"- **Language tagged**: {stats.get('lang_tagged', 'N/A'):,}",
            f"- **Anonymized fields**: {stats.get('anonymized', 'N/A'):,}",
        ])
        anon = stats.get("anon_entity_counts", {})
        if anon:
            lines.append("")
            lines.append("### Anonymization Breakdown")
            lines.append("")
            for etype, cnt in sorted(anon.items(), key=lambda x: -x[1]):
                lines.append(f"- `{etype}`: {cnt:,}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            load_dotenv()
            token = os.environ.get("HF_TOKEN")
        except ImportError:
            pass
    if not token:
        print("Error: HF_TOKEN not set")
        return

    files = sorted(PROCESSED_DIR.glob("*.json"))
    print(f"Found {len(files)} processed files")

    rows: list[dict] = []
    skipped = 0
    for f in tqdm(files, desc="Reading"):
        try:
            data = orjson.loads(f.read_bytes())
        except Exception:
            skipped += 1
            continue
        row = flatten_post(data)
        if row:
            rows.append(row)
        else:
            skipped += 1

    print(f"Loaded {len(rows)} posts, skipped {skipped}")

    df = pd.DataFrame(rows)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Write parquet to buffer
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()
    print(f"Parquet size: {len(parquet_bytes) / 1024 / 1024:.1f} MB")

    # Load processing stats for README
    stats = None
    if STATS_PATH.exists():
        stats = orjson.loads(STATS_PATH.read_bytes())

    readme = build_readme(stats, len(rows))

    # Upload atomically
    api = HfApi(token=token)
    operations = [
        CommitOperationAdd(
            path_in_repo="data/train-00000-of-00001.parquet",
            path_or_fileobj=io.BytesIO(parquet_bytes),
        ),
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=readme.encode(),
        ),
    ]

    print(f"Uploading to {REPO_ID}...")
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Update dataset ({len(rows):,} posts)",
    )
    print("Done!")


if __name__ == "__main__":
    main()
