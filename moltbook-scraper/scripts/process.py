from __future__ import annotations

import argparse
import html
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import orjson
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Globals set by worker initializer
# ---------------------------------------------------------------------------
_ft_model = None
_analyzer = None
_anonymizer = None
_blocklist: set[str] = set()
_bip39_words: set[str] = set()

REMOVED_SPAM = "<REMOVED-SPAM>"
REMOVED_BLOCKLIST = "<REMOVED-BLOCKLIST>"
REMOVED_TOO_LONG = "<REMOVED-TOO-LONG>"

FASTTEXT_HF_REPO = "facebook/fasttext-language-identification"
FASTTEXT_HF_FILE = "model.bin"
BIP39_URL = "https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt"
BIP39_PATH = Path("data/metadata/bip39_english.txt")

PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "CRYPTO",
    "IBAN_CODE",
    "US_SSN",
    "US_ITIN",
    "API_KEY",
    "PASSWORD",
    "SEED_PHRASE",
]


# ---------------------------------------------------------------------------
# Custom Presidio recognizers
# ---------------------------------------------------------------------------
def _build_custom_recognizers() -> list:
    from presidio_analyzer import Pattern, PatternRecognizer

    api_key = PatternRecognizer(
        supported_entity="API_KEY",
        patterns=[Pattern("api_key", r"sk-[A-Za-z0-9_\-]{20,100}", 0.9)],
    )
    password = PatternRecognizer(
        supported_entity="PASSWORD",
        patterns=[
            Pattern(
                "password_in_text",
                r"(?:password|passwd|pwd)\s*[:=]\s*\S+",
                0.8,
            )
        ],
    )
    return [api_key, password, _build_bip39_recognizer()]


def _build_bip39_recognizer():  # noqa: ANN202
    from presidio_analyzer import EntityRecognizer, RecognizerResult

    bip39 = _load_bip39_words()

    class Bip39Recognizer(EntityRecognizer):
        def load(self) -> None:
            pass

        def analyze(
            self,
            text: str,
            entities: list[str],
            nlp_artifacts=None,  # noqa: ANN001
        ) -> list[RecognizerResult]:
            results: list[RecognizerResult] = []
            words = text.lower().split()
            if len(words) < 12:
                return results

            run_start: int | None = None
            run_len = 0
            for i, w in enumerate(words):
                if w in bip39:
                    if run_start is None:
                        run_start = i
                    run_len += 1
                else:
                    if run_len >= 12 and run_start is not None:
                        s = _word_index_to_char(text, run_start)
                        e = _word_index_to_char_end(text, run_start + run_len - 1)
                        results.append(
                            RecognizerResult(
                                entity_type="SEED_PHRASE",
                                start=s,
                                end=e,
                                score=0.85,
                            )
                        )
                    run_start = None
                    run_len = 0

            if run_len >= 12 and run_start is not None:
                s = _word_index_to_char(text, run_start)
                e = _word_index_to_char_end(text, run_start + run_len - 1)
                results.append(
                    RecognizerResult(
                        entity_type="SEED_PHRASE",
                        start=s,
                        end=e,
                        score=0.85,
                    )
                )
            return results

    return Bip39Recognizer(supported_entities=["SEED_PHRASE"], name="BIP39Recognizer")


def _word_index_to_char(text: str, word_idx: int) -> int:
    """Return char offset of the start of the word_idx-th whitespace-split word."""
    pos = 0
    for _ in range(word_idx):
        pos = text.index(" ", pos) + 1
    return pos


def _word_index_to_char_end(text: str, word_idx: int) -> int:
    """Return char offset past the end of the word_idx-th word."""
    start = _word_index_to_char(text, word_idx)
    space = text.find(" ", start)
    return space if space != -1 else len(text)


# ---------------------------------------------------------------------------
# Worker initializer
# ---------------------------------------------------------------------------
def _load_bip39_words() -> set[str]:
    return {
        line.strip().lower()
        for line in BIP39_PATH.read_text().splitlines()
        if line.strip()
    }


def _worker_init(blocklist_file: str | None) -> None:
    global _ft_model, _analyzer, _anonymizer, _blocklist, _bip39_words  # noqa: PLW0603

    import fasttext
    import fasttext.FastText as _ft_mod
    import numpy as np

    # Patch NumPy 2.x incompatibility in fasttext's predict().
    # fasttext calls np.array(probs, copy=False) which raises in NumPy 2.x
    # when a copy is needed. Replace with a wrapper that drops the copy kwarg.
    _real_np_array = np.array

    def _patched_array(*a, **kw):  # noqa: ANN002, ANN003, ANN202
        kw.pop("copy", None)
        return _real_np_array(*a, **kw)

    _ft_mod.np.array = _patched_array  # type: ignore[attr-defined]

    _ft_model = fasttext.load_model(
        hf_hub_download(
            repo_id="facebook/fasttext-language-identification", filename="model.bin"
        )
    )

    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    _analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
    for rec in _build_custom_recognizers():
        _analyzer.registry.add_recognizer(rec)
    _anonymizer = AnonymizerEngine()

    _bip39_words = _load_bip39_words()

    if blocklist_file and Path(blocklist_file).exists():
        _blocklist = {
            line.strip().lower()
            for line in Path(blocklist_file).read_text().splitlines()
            if line.strip()
        }


# ---------------------------------------------------------------------------
# Content traversal
# ---------------------------------------------------------------------------
def iter_content_fields(
    record: dict,
) -> list[tuple[dict, str, str, str]]:
    """Return list of (parent_dict, key, entity_id, post_id) for each text field."""
    fields: list[tuple[dict, str, str, str]] = []
    post = record.get("post", {})
    post_id = post.get("id", "unknown")
    if post.get("title"):
        fields.append((post, "title", post_id, post_id))
    if post.get("content"):
        fields.append((post, "content", post_id, post_id))

    for comment in record.get("comments", []):
        _collect_comment_fields(comment, post_id, fields)
    return fields


def _collect_comment_fields(
    comment: dict, post_id: str, fields: list[tuple[dict, str, str, str]]
) -> None:
    comment_id = comment.get("id", "unknown")
    if comment.get("content"):
        fields.append((comment, "content", comment_id, post_id))
    for reply in comment.get("replies", []):
        _collect_comment_fields(reply, post_id, fields)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def is_spam(text: str) -> bool:
    tokens = text.lower().split()
    n = len(tokens)
    if n < 10:
        return False
    if len(set(tokens)) == 1:
        return True
    for phrase_len in (2, 3):
        if n >= phrase_len:
            phrase = tuple(tokens[:phrase_len])
            reps = 0
            for i in range(0, n - phrase_len + 1, phrase_len):
                if tuple(tokens[i : i + phrase_len]) == phrase:
                    reps += 1
                else:
                    break
            if reps >= 6:
                return True
    return len(set(tokens)) <= 2 and n >= 10


def check_blocklist(text_lower: str) -> bool:
    return any(phrase in text_lower for phrase in _blocklist)


def detect_language(text: str) -> tuple[str, float]:
    labels, scores = _ft_model.predict(text.replace("\n", " "), k=1)
    lang = labels[0].replace("__label__", "")
    return lang, float(scores[0])


def anonymize_text(text: str) -> tuple[str, dict[str, int]]:
    """Return (anonymized_text, entity_type_counts)."""
    results = _analyzer.analyze(
        text=text,
        entities=PRESIDIO_ENTITIES,
        language="en",
    )
    if not results:
        return text, {}
    counts: dict[str, int] = {}
    for r in results:
        counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
    anon = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anon.text, counts


def process_field(text: str, max_tokens: int) -> tuple[str, dict[str, Any]]:
    """Run the full pipeline on a single text field. Return (new_text, field_stats)."""
    stats: dict[str, Any] = {
        "normalized_changed": False,
        "removed_spam": False,
        "removed_blocklist": False,
        "removed_too_long": False,
        "lang": None,
        "lang_score": None,
        "anonymized": False,
        "anon_entity_counts": {},
    }

    # 1) Normalize
    normed = normalize(text)
    if normed != text:
        stats["normalized_changed"] = True
    text = normed

    # 2) Spam
    if is_spam(text):
        stats["removed_spam"] = True
        return REMOVED_SPAM, stats

    # 3) Blocklist
    if _blocklist and check_blocklist(text.lower()):
        stats["removed_blocklist"] = True
        return REMOVED_BLOCKLIST, stats

    # 4) Token length
    if len(text.split()) > max_tokens:
        stats["removed_too_long"] = True
        return REMOVED_TOO_LONG, stats

    # 5) Language
    lang, lang_score = detect_language(text)
    stats["lang"] = lang
    stats["lang_score"] = lang_score

    # 6) Anonymize
    anon, anon_counts = anonymize_text(text)
    if anon_counts:
        stats["anonymized"] = True
        stats["anon_entity_counts"] = anon_counts
    text = anon

    return text, stats


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------
def process_file(
    args: tuple[Path, Path, int, bool],
) -> dict[str, Any]:
    input_path, output_path, max_tokens, overwrite = args

    file_stats: dict[str, Any] = {
        "file": input_path.name,
        "total_fields": 0,
        "normalized_changed": 0,
        "removed_spam": 0,
        "removed_blocklist": 0,
        "removed_too_long": 0,
        "lang_tagged": 0,
        "anonymized": 0,
        "anon_entity_counts": {},
        "spam_ids": [],
        "blocklist_ids": [],
        "long_content_ids": [],
        "error": None,
    }

    if not overwrite and output_path.exists():
        file_stats["skipped"] = True
        return file_stats

    try:
        data = orjson.loads(input_path.read_bytes())
    except Exception as e:
        file_stats["error"] = str(e)
        return file_stats

    for parent, key, entity_id, post_id in iter_content_fields(data):
        file_stats["total_fields"] += 1
        text = parent[key]
        if not isinstance(text, str) or not text.strip():
            continue

        new_text, field_stats = process_field(text, max_tokens)
        parent[key] = new_text

        hit = {"post_id": post_id, "entity_id": entity_id, "field": key}
        if field_stats["normalized_changed"]:
            file_stats["normalized_changed"] += 1
        if field_stats["removed_spam"]:
            file_stats["removed_spam"] += 1
            file_stats["spam_ids"].append(hit)
        if field_stats["removed_blocklist"]:
            file_stats["removed_blocklist"] += 1
            file_stats["blocklist_ids"].append(hit)
        if field_stats["removed_too_long"]:
            file_stats["removed_too_long"] += 1
            file_stats["long_content_ids"].append(hit)
        if field_stats["lang"] is not None:
            file_stats["lang_tagged"] += 1
            parent["lang"] = field_stats["lang"]
            parent["lang_score"] = field_stats["lang_score"]
        if field_stats["anonymized"]:
            file_stats["anonymized"] += 1
            for etype, cnt in field_stats["anon_entity_counts"].items():
                file_stats["anon_entity_counts"][etype] = (
                    file_stats["anon_entity_counts"].get(etype, 0) + cnt
                )

    # Atomic write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    tmp_path.rename(output_path)

    return file_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw Moltbook JSON files.")
    parser.add_argument("--input-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="data/processed", type=Path)
    parser.add_argument("--max-tokens", default=100_000, type=int)
    parser.add_argument("--blocklist-file", default="data/metadata/filter_words.txt")
    parser.add_argument("--max-workers", default=3, type=int)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--stats-file",
        default="data/metadata/processing_stats.json",
        type=Path,
    )
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.json"))
    print(f"Found {len(input_files)} input files")

    work_items = [
        (f, args.output_dir / f.name, args.max_tokens, args.overwrite)
        for f in input_files
    ]
    # work_items = work_items[:100_000]

    agg: dict[str, Any] = {
        "total_files": len(input_files),
        "total_fields": 0,
        "normalized_changed": 0,
        "removed_spam": 0,
        "removed_blocklist": 0,
        "removed_too_long": 0,
        "lang_tagged": 0,
        "anonymized": 0,
        "anon_entity_counts": {},
        "skipped": 0,
        "errors": 0,
        "spam_ids": [],
        "blocklist_ids": [],
        "long_content_ids": [],
    }

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=_worker_init,
        initargs=(args.blocklist_file,),
    ) as executor:
        for fs in tqdm(
            executor.map(process_file, work_items, chunksize=64),
            total=len(work_items),
            desc="Processing",
        ):
            if fs.get("skipped"):
                agg["skipped"] += 1
                continue
            if fs.get("error"):
                agg["errors"] += 1
                continue
            agg["total_fields"] += fs["total_fields"]
            agg["normalized_changed"] += fs["normalized_changed"]
            agg["removed_spam"] += fs["removed_spam"]
            agg["removed_blocklist"] += fs["removed_blocklist"]
            agg["removed_too_long"] += fs["removed_too_long"]
            agg["lang_tagged"] += fs["lang_tagged"]
            agg["anonymized"] += fs["anonymized"]
            for etype, cnt in fs["anon_entity_counts"].items():
                agg["anon_entity_counts"][etype] = (
                    agg["anon_entity_counts"].get(etype, 0) + cnt
                )
            for k in ("spam_ids", "blocklist_ids", "long_content_ids"):
                agg[k].extend(fs[k])

    # Print summary
    processed = agg["total_files"] - agg["skipped"] - agg["errors"]
    print("\n--- Processing Summary ---")
    print(f"Files processed: {processed}")
    print(f"Files skipped (already exist): {agg['skipped']}")
    print(f"Files with errors: {agg['errors']}")
    print(f"Total content fields: {agg['total_fields']}")
    print(f"Normalized (changed): {agg['normalized_changed']}")
    print(f"Removed (spam): {agg['removed_spam']}")
    print(f"Removed (blocklist): {agg['removed_blocklist']}")
    print(f"Removed (too long): {agg['removed_too_long']}")
    print(f"Language tagged: {agg['lang_tagged']}")
    print(f"Anonymized: {agg['anonymized']}")
    if agg["anon_entity_counts"]:
        print("  Entity types anonymized:")
        for etype, cnt in sorted(
            agg["anon_entity_counts"].items(), key=lambda x: -x[1]
        ):
            print(f"    {etype}: {cnt}")

    # Write stats
    stats_path = args.stats_file
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_bytes(orjson.dumps(agg, option=orjson.OPT_INDENT_2))
    print(f"\nStats written to {stats_path}")


if __name__ == "__main__":
    main()
