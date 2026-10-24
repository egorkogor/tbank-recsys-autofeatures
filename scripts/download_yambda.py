"""Download Yambda-50M (flat) + item mappings (+ optional audio embeddings) from Hugging Face."""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "yambda"
FILES = [f"flat/50m/{n}.parquet" for n in ["listens", "likes", "dislikes", "unlikes", "undislikes"]]
FILES += ["artist_item_mapping.parquet", "album_item_mapping.parquet"]

if __name__ == "__main__":
    files = FILES + (["embeddings.parquet"] if "--embeddings" in sys.argv else [])
    if "--only-embeddings" in sys.argv:
        files = ["embeddings.parquet"]
    for f in files:
        p = hf_hub_download("yandex/yambda", f, repo_type="dataset", local_dir=OUT)
        print("ok", p, flush=True)
