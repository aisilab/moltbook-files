import json
import time
from pathlib import Path

import httpx

QUEUE_FILE = Path("data/metadata/posts_queue.jsonl")
LIMIT = 150
SLEEP_SECONDS = 1.0


def load_existing_ids() -> set[str]:
    """Load all existing post IDs from the queue file."""
    if not QUEUE_FILE.exists():
        return set()
    ids = set()
    with open(QUEUE_FILE) as f:
        for line in f:
            if line.strip():
                post = json.loads(line)
                ids.add(post["id"])
    return ids


def append_post_to_queue(post: dict) -> None:
    """Append a single post to the JSONL queue file."""
    with open(QUEUE_FILE, "a") as f:
        f.write(json.dumps(post) + "\n")


HEADERS = {
    "accept": "*/*",
    "accept-language": "en-GB,en;q=0.9",
    "priority": "u=1, i",
    "referer": "https://www.moltbook.com/",
    "sec-ch-ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
}


def fetch_posts_for_sort(sort_params: dict, existing_ids: set[str]) -> tuple[int, int]:
    sort_modes = [
        {**sort_params, "sort": "new"},
        {**sort_params, "sort": "top", "time": "all"},
        {**sort_params, "sort": "comments", "time": "all"},
        {**sort_params, "sort": "top", "time": "day"},
        {**sort_params, "sort": "comments", "time": "day"},
    ]

    total_added = 0
    total_skipped = 0
    offset = 0

    while True:
        print(f"\n--- offset={offset} ---")
        any_had_posts = False

        for mode in sort_modes:
            label = "&".join(f"{k}={v}" for k, v in mode.items())

            params = {"limit": LIMIT, "offset": offset, **mode}
            response = httpx.get(
                "https://www.moltbook.com/api/v1/posts",
                params=params,
                headers=HEADERS,
                timeout=120.0,
            )

            data = response.json()

            if not data.get("success") or not data.get("posts"):
                print(f"  {label}: no posts")
                continue

            any_had_posts = True
            added = 0
            skipped = 0
            for post in data["posts"]:
                if post["id"] in existing_ids:
                    skipped += 1
                else:
                    append_post_to_queue(post)
                    existing_ids.add(post["id"])
                    added += 1

            total_added += added
            total_skipped += skipped
            print(f"  {label}: +{added}, skipped {skipped} (total: {total_added})")
            time.sleep(SLEEP_SECONDS)

        if not any_had_posts:
            print("All sort modes exhausted.")
            break

        offset += LIMIT

    return total_added, total_skipped


def fetch_all_posts() -> None:
    existing_ids = load_existing_ids()
    print(f"Loaded {len(existing_ids)} existing post IDs")

    added, skipped = fetch_posts_for_sort({}, existing_ids)
    print(f"\nDone! Added {added} new posts, skipped {skipped} duplicates")


if __name__ == "__main__":
    fetch_all_posts()
