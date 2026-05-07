import pandas as pd
from datasets import load_dataset

name = "aisilab/moltbook-files"
dataset = load_dataset(name)
print(dataset)
# to pandas
df = pd.DataFrame(dataset["train"])
df = df[df["lang"] == "lit_Latn"]
print(df["lang_score"].describe())

# drop duplicates based on the 'content'
df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
df.to_json("lit.json", orient="records", lines=True)
