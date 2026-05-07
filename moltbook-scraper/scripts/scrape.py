import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx

BATCH_SIZE = 64
DATA_DIR = Path("data/raw")
QUEUE_FILE = Path("data/metadata/posts_queue.jsonl")
FAILURES_FILE = Path("data/metadata/fetch_failures.jsonl")

HEADERS = {
    "accept": "*/*",
    "accept-language": "en-GB,en;q=0.9",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
}


def load_failed_ids() -> set[str]:
    if not FAILURES_FILE.exists():
        return set()
    ids = set()
    with open(FAILURES_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            ids.add(json.loads(line)["id"])
    return ids


def load_pending_posts(limit: int, failed_ids: set[str]) -> list[dict]:
    if not QUEUE_FILE.exists():
        return []

    DATA_DIR.mkdir(exist_ok=True)
    pending = []

    with open(QUEUE_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            post = json.loads(line)
            post_id = post["id"]
            if post_id in failed_ids:
                continue
            json_path = DATA_DIR / f"{post_id}.json"
            if not json_path.exists() or json_path.stat().st_size == 0:
                pending.append(post)
                if len(pending) >= limit:
                    break

    return pending


def log_failure(post_id: str, error: str) -> None:
    record = {
        "id": post_id,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    with open(FAILURES_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


async def fetch_post(
    client: httpx.AsyncClient, post_id: str
) -> tuple[bool, str | None]:
    url = f"https://www.moltbook.com/api/v1/posts/{post_id}"
    referer = f"https://www.moltbook.com/post/{post_id}"

    try:
        response = await client.get(url, headers={"referer": referer})
        response.raise_for_status()
        data = response.json()
        DATA_DIR.mkdir(exist_ok=True)
        (DATA_DIR / f"{post_id}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        return True, None
    except httpx.HTTPStatusError as e:
        return False, f"HTTP {e.response.status_code}"
    except Exception as e:
        return False, str(e)


async def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    failed_ids = load_failed_ids()
    print(f"Skipping {len(failed_ids)} previously failed posts")

    total_success = 0
    total_failure = 0

    async with httpx.AsyncClient(
        headers=HEADERS, timeout=60.0, follow_redirects=True
    ) as client:
        while True:
            pending = load_pending_posts(BATCH_SIZE, failed_ids)
            if not pending:
                print("No more pending posts to fetch.")
                break

            print(f"Processing batch of {len(pending)} posts...")

            tasks = [fetch_post(client, post["id"]) for post in pending]
            results = await asyncio.gather(*tasks)

            for post, (success, error) in zip(pending, results, strict=False):
                post_id = post["id"]
                if success:
                    total_success += 1
                    print(f"  ✓ {post_id}")
                else:
                    total_failure += 1
                    failed_ids.add(post_id)
                    log_failure(post_id, error or "Unknown error")
                    print(f"  ✗ {post_id}: {error}")

            print(
                f"  Batch complete. Total: {total_success} success, {total_failure} failed"
            )
            await asyncio.sleep(1)

    print(f"\nDone! Fetched {total_success} posts, {total_failure} failures")


if __name__ == "__main__":
    asyncio.run(main())
