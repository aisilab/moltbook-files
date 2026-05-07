import json
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import spacy
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from wordcloud import WordCloud

enc = tiktoken.encoding_for_model("gpt-4o")
name = "aisilab/moltbook-files"
dataset = load_dataset(name)
# to pandas 
df = pd.DataFrame(dataset["train"])

ids_to_remove = ["693c52b7-c3b3-4ac5-bfe2-f2166e40a021"] # too long
df = df[~df['post_id'].isin(ids_to_remove)].reset_index(drop=True)

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

def build_content(row) -> str:
    lines = [
        f"/{row['submolt_name']}/{row['post_id']}",
        f"{row['title']}",
        f"{row['author_name']}",
        f"{row['content']}",
        "",
    ]
    comments = flatten_comments(row["comments"])
    for comment in comments:
        lines.append(f"  # {comment['author_name']}: {comment['content']}")
    return "\n".join(lines)

def plot_token_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    # keep only tokens < 200_000
    thresh_hold = 5_000
    tokens = df[df['tokens'] < thresh_hold]
    # how many % of posts are below thresh_hold
    percent_below = len(tokens) / len(df) * 100
    plt.hist(tokens['tokens'], bins=20, color='blue', alpha=0.7)
    plt.title(f'Token Count Distribution (< {thresh_hold} tokens) ({percent_below:.2f}%)')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Posts')
    plt.grid(True)
    # Save the plot
    plt.savefig('data/plots/token_distribution.png')
    #plt.show()


def plot_word_cloud(df: pd.DataFrame) -> None:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    counts = Counter()
    df = df.copy()
    df = df[df['tokens'] < 5_000]  # filter out very long posts to avoid memory issues
    for doc in tqdm(nlp.pipe(df['content'], batch_size=16), total=len(df), desc="Lemmatizing"):
        counts.update(
            token.lemma_.lower() for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) > 1
        )
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(counts)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.savefig('data/plots/word_cloud.png')
    #plt.show()

def count_comments(comments: str | list[dict]) -> int:
    if isinstance(comments, str):
        comments = json.loads(comments)
    total = 0
    for comment in comments:
        total += 1
        if comment.get("replies"):
            total += count_comments(comment["replies"])
    return total


def get_comment_langs(comments: str | list[dict]) -> list[str]:
    """Extract all lang values from comments (including nested replies)."""
    if isinstance(comments, str):
        comments = json.loads(comments)
    langs = []
    for comment in comments:
        if comment.get('lang'):
            langs.append(comment['lang'])
        if comment.get('replies'):
            langs.extend(get_comment_langs(comment['replies']))
    return langs


def parse_lang(lang: str) -> str:
    """Extract language code from format like '__label__eng_Latn' -> 'eng'."""
    if not lang:
        return lang
    # Handle formats like '__label__eng_Latn' or '_label_eng_Latn'
    parts = lang.replace('__', '_').strip('_').split('_')
    # parts: ['label', 'eng', 'Latn'] -> return 'eng'
    if len(parts) >= 2 and parts[0] == 'label':
        return parts[1]
    return lang



def plot_lang_distribution_posts(df: pd.DataFrame) -> None:
    lang_counts = df['lang'].value_counts()
    lang_counts = lang_counts.head(20)
    labels = [lang for lang in lang_counts.index]
    # use only top 20 langs
    plt.figure(figsize=(10, 6))
    plt.bar(labels, lang_counts.values, color='blue', alpha=0.7)
    plt.title('Language Distribution (Posts)')
    plt.xlabel('Language')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/plots/lang_distribution_posts.png')
    #plt.show()


def plot_lang_distribution_overall(df: pd.DataFrame) -> None:
    # Collect post languages
    all_langs = df['lang'].tolist()
    # Collect comment languages
    for comments in df['comments']:
        all_langs.extend(get_comment_langs(comments))
    lang_counts = pd.Series(all_langs).value_counts()
    lang_counts = lang_counts.head(20)
    labels = [lang for lang in lang_counts.index]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, lang_counts.values, color='purple', alpha=0.7)
    plt.title('Language Distribution (Overall: Posts + Comments)')
    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/plots/lang_distribution_overall.png')
    #plt.show()  


def plot_time_distribution(df: pd.DataFrame) -> None:
    df_time = df.copy()
    df_time['created_at'] = pd.to_datetime(df_time['created_at'])
    df_time['date'] = df_time['created_at'].dt.date

    posts_per_day = df_time.groupby('date').size()
    plt.figure(figsize=(10, 6))
    plt.bar(posts_per_day.index, posts_per_day.values, color='blue', alpha=0.7)
    plt.title('Posts per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/plots/time_distribution.png')
    #plt.show()


def print_schema(df: pd.DataFrame) -> None:
    """Print schema for posts and comments."""
    print("=== Post Schema ===")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"  {col}: {dtype}")

    # Get a sample comment to show comment schema
    print("\n=== Comment Schema ===")
    sample_comments = df['comments'].iloc[9]
    if isinstance(sample_comments, str):
        sample_comments = json.loads(sample_comments)
    if sample_comments:
        comment = sample_comments[0]
        for key, value in comment.items():
            dtype = type(value).__name__
            print(f"  {key}: {dtype}")


def save_stats_csv(df: pd.DataFrame) -> None:
    total_posts = len(df)
    total_comments = df['comments'].apply(count_comments).sum()
    unique_communities = df['submolt_name'].nunique()
    unique_authors = df['author_name'].nunique()
    date_min = df['created_at'].min()[:10]
    date_max = df['created_at'].max()[:10]

    stats = pd.DataFrame({
        'Metric': ['Total Posts', 'Total Comments', 'Unique Communities', 'Unique Authors', 'Date Range'],
        'Value': [total_posts, total_comments, unique_communities, unique_authors, f"{date_min} to {date_max}"]
    })
    stats.to_csv('data/plots/stats.csv', index=False)
    print(stats)


print_schema(df)

df['content'] = df.apply(build_content, axis=1)
df['tokens'] = df['content'].apply(lambda x: len(enc.encode(x, disallowed_special=())))

# plot_token_distribution(df)
# try:
#     plot_word_cloud(df)
# except Exception as e:
#     print(f"Error generating word cloud: {e}")
plot_time_distribution(df)
save_stats_csv(df)
plot_lang_distribution_posts(df)
plot_lang_distribution_overall(df)

