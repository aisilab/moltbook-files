"""Upload a folder to a Hub model repo (public, like the existing filter-with-espresso adapters).

The write token comes from HF_TOKEN.
Usage:  python push.py <folder> <repo_id> [path_in_repo]
"""

import sys

from huggingface_hub import HfApi

folder, repo = sys.argv[1], sys.argv[2]
path_in_repo = sys.argv[3] if len(sys.argv) > 3 else ""

api = HfApi()
api.create_repo(repo, private=False, exist_ok=True)
api.upload_folder(folder_path=folder, repo_id=repo, path_in_repo=path_in_repo)
print(f"pushed {folder} -> {repo}/{path_in_repo}")
