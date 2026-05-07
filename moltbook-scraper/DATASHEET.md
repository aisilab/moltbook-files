# Datasheet for the Moltbook Files Dataset

Following the *Datasheets for Datasets* template (Gebru et al., 2018, [arXiv:1803.09010](https://arxiv.org/abs/1803.09010)). The dataset itself is hosted at [`aisilab/moltbook-files`](https://huggingface.co/datasets/aisilab/moltbook-files); this document describes its provenance, composition, and intended use. The code that produced it is in this repository.

## Motivation

**For what purpose was the dataset created?**
To enable research on social-media discourse on the Moltbook platform — topic modelling, language identification at scale, multilingual NLP, content moderation studies, and analysis of community structure. There was no prior public dump of Moltbook content.

**Who created the dataset and on behalf of which entity?**
The dataset was assembled by the authors of this repository as part of their academic research.

**Who funded the creation of the dataset?**
Self-funded; no external grant.

## Composition

**What do the instances represent?**
Each instance is a Moltbook *post* with its full *comment thread*. A post belongs to a *submolt* (community), has a title, body, language tag, vote counts, and timestamp. Comments share the same shape but are nested via `parent_id` and recursive `replies`.

**How many instances are there in total?**
- **232,497** posts
- **2,202,950** comments
- **3,628** unique communities (submolts)
- **34,905** unique authors
- Date range of `created_at`: **2026-01-27 to 2026-02-07** (the snapshot window during which posts were observable via Moltbook's public API; the API exposes new/top/comments sorts that surfaced posts in this window).

**Does the dataset contain all possible instances or is it a sample?**
A sample. Posts were discovered via Moltbook's public API (`/api/v1/posts`) using five sort strategies (`new`, `top all-time`, `top day`, `comments all-time`, `comments day`) paginated until exhaustion. Posts older than the API's reachable horizon, posts in private submolts, and deleted posts are not included. Posts that 404'd at scrape time are logged in `data/metadata/fetch_failures.jsonl` and excluded.

**What data does each instance consist of?**
Schema (see also `scripts/push_dataset.py:71-128`):

| Column | Type | Description |
|---|---|---|
| `post_id` | string (UUID) | unique post identifier |
| `title` | string | post title (PII-anonymized) |
| `content` | string | post body (PII-anonymized) |
| `url` | string | original Moltbook URL |
| `upvotes`, `downvotes`, `comment_count` | int | engagement metrics |
| `created_at` | string (ISO 8601) | post timestamp |
| `submolt_id`, `submolt_name` | string | community |
| `author_id`, `author_name` | string | post author (see Privacy below) |
| `lang` | string | fastText language code (e.g. `eng_Latn`) |
| `lang_score` | float | fastText confidence |
| `comments` | string (JSON) | nested comment tree |

**Are there any errors, sources of noise, or redundancies?**
- Spam posts (46 detected) and blocklist hits (91) are replaced with `<REMOVED-SPAM>` / `<REMOVED-BLOCKLIST>` markers in `content`/`title`, not deleted, so the row count and engagement metrics remain comparable. See `scripts/process.py:237-260`.
- fastText language ID is occasionally wrong on very short posts; `lang_score` is provided so users can filter.
- Some posts are crossposts; we do not deduplicate across communities.

**Is the dataset self-contained or does it rely on external resources?**
Self-contained as a parquet on HuggingFace. The pipeline that produced it relies on `facebook/fasttext-language-identification` (auto-downloaded), the BIP-39 English wordlist (downloaded once into `data/metadata/bip39_english.txt`), and spaCy's `en_core_web_sm` model.

**Does the dataset contain data that might be considered confidential, offensive, or upsetting?**
Yes — Moltbook is a general-purpose social platform and posts include user-generated text that may be offensive, sensitive, or NSFW. We apply automated PII redaction (see Preprocessing) but do not filter for opinion, slurs, or graphic content. Researchers should treat the corpus as raw web text.

## Collection Process

**How was the data acquired?**
HTTP GET requests to the Moltbook public REST API. Two stages, in this order:
1. `scripts/crawl.py` paginates `/api/v1/posts` and accumulates post IDs into `data/metadata/posts_queue.jsonl`. One request per second per sort mode (`SLEEP_SECONDS = 1.0`).
2. `scripts/scrape.py` async-fetches `/api/v1/posts/{id}` for each queued ID, 64 concurrent requests per batch, 1 s pause between batches.

Both scripts use a standard browser User-Agent and the `Referer: https://www.moltbook.com/` header — Moltbook serves the same JSON to its own front-end.

**Over what timeframe was the data collected?**
The crawl was performed in February 2026; the snapshot used in this repository corresponds to commit `<SET BEFORE SUBMISSION>` of `aisilab/moltbook-files`.

**Does the dataset relate to people?**
Yes. Moltbook is a public social platform. Posts and comments are written by individuals, and `author_name` / `comment.author_name` are public-display usernames as exposed by the Moltbook API. We do not collect emails, profile photos, follower graphs, IPs, or private messages.

**Were the individuals notified about the data collection?**
No individual notification. The data was collected from a public API in the same manner as Moltbook's own web client. Users were not asked to consent specifically to research use. See *Uses* and *Maintenance* below for opt-out mechanism.

**Was an ethics review process conducted?**
The collection was reviewed against the principles of common practice for public-API social-media research. Since the data is publicly visible to anyone visiting Moltbook, and we apply automated PII redaction to the textual content, we deemed the collection low risk. No formal IRB review was performed.

## Preprocessing / Cleaning / Labelling

Implemented in `scripts/process.py`. Per text field (post title, post content, every comment content), in order:

1. **Normalize** — HTML unescape + collapse whitespace.
2. **Spam detect** — heuristic: replicated 2-/3-grams, mono-token spam (lines 237–256). Hits → `<REMOVED-SPAM>`. **46 fields removed.**
3. **Blocklist** — phrase match against `data/metadata/filter_words.txt`. Hits → `<REMOVED-BLOCKLIST>`. **91 fields removed.**
4. **Length cap** — fields longer than 100,000 whitespace tokens → `<REMOVED-TOO-LONG>`. (None removed at this threshold in the published snapshot.)
5. **Language ID** — fastText `lid.176`. Tags every field with `lang` + `lang_score`. **2,663,740 / 2,663,967 fields tagged.**
6. **PII anonymization** — Microsoft Presidio with custom recognizers:

   | Entity | Detected count |
   |---|---|
   | `CRYPTO` (wallet addresses) | 7,203 |
   | `PHONE_NUMBER` | 3,240 |
   | `EMAIL_ADDRESS` | 2,176 |
   | `US_SSN` | 541 |
   | `PASSWORD` | 140 |
   | `API_KEY` (sk- prefix) | 48 |
   | `US_ITIN` | 14 |
   | `SEED_PHRASE` (≥12 BIP-39 words) | 7 |
   | `CREDIT_CARD` | 2 |
   | `IBAN_CODE` | 2 |
   | **Total fields anonymized** | **12,435** |

   Detected spans are replaced with `<ENTITY_TYPE>` placeholders.

The raw, pre-processing JSON files (`data/raw/`) are not redistributed. Re-creating them requires re-running the crawl + scrape against Moltbook's live API.

## Uses

**What tasks could the dataset be used for?**
- Multilingual topic modelling and clustering.
- Language identification benchmarks (especially long-tail languages — see `scripts/extras/lit_check.py` and `scripts/extras/new_langs.py`).
- Studying social-media discourse, community detection, comment-thread structure.
- Content moderation research (the labelled `<REMOVED-SPAM>` / `<REMOVED-BLOCKLIST>` markers are a small but signal-rich subset).

**Are there tasks for which the dataset should not be used?**
- Identifying or profiling individual users. Although `author_name` is retained (these are public-display handles), do not use this data to build user-level profiles, surveillance models, or harassment tooling.
- Training a model that re-emits PII verbatim. Even with our redaction, no automated PII filter is perfect.
- Any commercial use that violates Moltbook's Terms of Service.

## Distribution

**How is the dataset distributed?**
As a single Parquet file on HuggingFace at `aisilab/moltbook-files`, accompanied by an auto-generated `README.md` summarising schema and processing stats.

**License**
Open Data Commons Attribution License (ODC-BY) v1.0. The accompanying code in this repository is MIT-licensed (see `LICENSE`).

**Have any third parties imposed IP-based or other restrictions?**
Moltbook's Terms of Service govern the source data. The published Parquet contains derived, processed, redacted text; users of the dataset must additionally comply with Moltbook's terms if they re-derive raw data from it.

## Maintenance

**Who maintains the dataset?**
William Brach (wibrach@gmail.com).

**Will the dataset be updated?**
We expect to refresh the snapshot periodically. New versions will appear as new commits on the HuggingFace repo; the `README.md` on the repo records the latest stats. The git tag in this repository corresponding to the paper submission is pinned to the matching dataset commit.

**If others want to extend / build on / contribute, is there a mechanism for them to do so?**
Open issues on the GitHub repository. Pull requests welcome.

**How can someone request that their data be removed?**
Email the maintainer with the affected `post_id` or `author_name`. We will remove the matching rows from the next published snapshot. There is no real-time deletion mechanism.
