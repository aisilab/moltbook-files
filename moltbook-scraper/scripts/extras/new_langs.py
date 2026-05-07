import json
import os
import re
from typing import Literal

import dspy
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
API_BASE = os.getenv("API_BASE")

lm = dspy.LM("openai/gpt-5-mini-fiit", api_key=API_KEY, api_base=API_BASE)
dspy.configure(lm=lm)


class ClassifyNewLanguage(dspy.Signature):
    """Classify whether a content is about a new language (constructed language, conlang, invented language, or new slang/dialect)."""

    content: str = dspy.InputField(
        desc="Content of the post"
    )
    is_new_language: bool = dspy.OutputField(
        desc="True if the post is about a new/constructed language, False otherwise"
    )
    why: str = dspy.OutputField(
        desc="Brief explanation for the classification"
    )

class ExtractLanguageMetadata(dspy.Signature):
    """Extract metadata about a new/constructed language from a post."""

    content: str = dspy.InputField(desc="Content of the post about a new language")
    language_name: str = dspy.OutputField(desc="Name of the language")
    reason: Literal["avoiding oversight", "token efficiency", "programming language", "spoken", "other"] = dspy.OutputField(
        desc="Why this language exists"
    )
    search_hints: list[str] = dspy.OutputField(
        desc="Distinctive keywords, syntax tokens, or phrases that could identify this language in other posts"
    )


name = "aisilab/moltbook-files"
dataset = load_dataset(name)["train"]

ids_to_remove = {"693c52b7-c3b3-4ac5-bfe2-f2166e40a021"}  # too long
dataset = dataset.filter(lambda row: row["post_id"] not in ids_to_remove)


def flatten_comments(comments: str | list[dict]) -> list[dict]:
    if isinstance(comments, str):
        comments = json.loads(comments)
    result = []
    for comment in comments:
        replies = comment.pop("replies", [])
        result.append(comment)
        if replies:
            result.extend(flatten_comments(replies))
    return result


def build_content(row: dict) -> dict:
    lines = [
        f"/{row['submolt_name']}/{row['post_id']}",
        f"{row['title']}",
        f"{row['author_name']}",
        f"{row['content']}",
        # "COMMENTS:",
        "",
    ]
    # comments = flatten_comments(row["comments"])
    # for comment in comments:
    #     lines.append(f"  # {comment['author_name']}: {comment['content']}")
    return {"content": "\n".join(lines)}


dataset = dataset.map(build_content)


pattern = re.compile(r"language|new language|slang", re.IGNORECASE)
subset = dataset.filter(lambda row: bool(pattern.search(row["content"])))

print(f"Total: {len(dataset)}, Regex matches: {len(subset)}")

classify = dspy.ChainOfThought(ClassifyNewLanguage)
examples = [dspy.Example(content=row["content"]).with_inputs("content") for row in subset]
results = classify.batch(examples, num_threads=8)

confirmed_ids = {}  # post_id -> reasoning
for row, result in zip(subset, results, strict=True):
    if result.is_new_language:
        confirmed_ids[row["post_id"]] = result.why
        print(f"  YES — {row['title']} ({row['post_id']}): {result.why[:80]}")

print(f"\nRegex matches: {len(subset)}, Confirmed new language posts: {len(confirmed_ids)}")

confirmed_rows = [row for row in subset if row["post_id"] in confirmed_ids]
extract = dspy.ChainOfThought(ExtractLanguageMetadata)
meta_examples = [dspy.Example(content=row["content"]).with_inputs("content") for row in confirmed_rows]
meta_results = extract.batch(meta_examples, num_threads=8)

metadata = {}
for row, result in zip(confirmed_rows, meta_results, strict=True):
    metadata[row["post_id"]] = {
        "language_name": result.language_name,
        "reason": result.reason,
        "search_hints": result.search_hints,
    }
    print(f"  META — {result.language_name} | {result.reason} | {result.search_hints}")

original = load_dataset(name)["train"]
new_langs_dataset = original.filter(lambda row: row["post_id"] in confirmed_ids)
new_langs_dataset = new_langs_dataset.map(lambda row: {
    "reasoning": confirmed_ids[row["post_id"]],
    "language_name": metadata[row["post_id"]]["language_name"],
    "reason": metadata[row["post_id"]]["reason"],
    "search_hints": metadata[row["post_id"]]["search_hints"],
})
new_langs_dataset.push_to_hub("aisilab/moltbook-files-new-language-signals")
print(f"Pushed {len(new_langs_dataset)} posts to HuggingFace")

