"""
Indexer that walks ./data/<dir>/<image>.jpg (e.g. ./data/0/0.jpg ... ./data/99/9999.jpg)
and bulk-indexes features into Elasticsearch.

Usage:
  python indexer.py --data-dir ./data --index images --batch 256 --op create --ipca-hog /models/ipca_hog.joblib --ipca-lbp /models/ipca_lbp.joblib

Notes:
  - Expects shared.features to provide VGGExtractor, HOGExtractor, LBPExtractor.
  - If PCA model paths are omitted, HOG/LBP will be used as-is (may be high-dim).
  - Uses Elasticsearch helpers.bulk; set --op create to skip already-created docs (safe resume).
"""
import os
import argparse
import math
import time
from pathlib import Path
import numpy as np
import joblib # type: ignore
from elasticsearch import Elasticsearch, helpers # type: ignore
from tqdm import tqdm # type: ignore
 
# Import your shared extractors (adjust module path if different)
from shared.feature_extractor import VGGExtractor, HOGExtractor, LBPExtractor

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default=os.getenv("DATA_ROOT", "/data"),
                    help="Root data directory containing numeric subfolders (0..99)")
parser.add_argument("--index", type=str, default=os.getenv("ES_INDEX", "images"),
                    help="Elasticsearch index name")
parser.add_argument("--es-host", type=str, default=os.getenv("ES_HOST", "http://elasticsearch:9200"),
                    help="Elasticsearch host URL")
parser.add_argument("--batch", type=int, default=int(os.getenv("BATCH", "128")),
                    help="Bulk batch size")
parser.add_argument("--op", choices=["index", "create"], default="index",
                    help="helpers.bulk op type: 'create' to skip existing docs, 'index' to overwrite")
parser.add_argument("--ipca-hog", type=str, default=os.getenv("IPCA_HOG", ""),
                    help="Path to IncrementalPCA joblib for HOG (optional)")
parser.add_argument("--ipca-lbp", type=str, default=os.getenv("IPCA_LBP", ""),
                    help="Path to IncrementalPCA joblib for LBP (optional)")
parser.add_argument("--start-dir", type=int, default=None,
                    help="Optional: numeric start directory to resume (inclusive)")
parser.add_argument("--end-dir", type=int, default=None,
                    help="Optional: numeric end directory to resume (inclusive)")
parser.add_argument("--ext", type=str, default=".jpg",
                    help="Image extension to look for (default .jpg)")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
INDEX_NAME = args.index
ES_HOST = args.es_host
BATCH = args.batch
OP = args.op
print(OP)
IPCA_HOG_PATH = args.ipca_hog or None
EXT = args.ext.lower()
TAGS_ROOT = Path(os.getenv("TAGS_ROOT", "/tags"))
# ---------- Init ----------
es = Elasticsearch(ES_HOST)

vgg = VGGExtractor()
hog = HOGExtractor()
lbp = LBPExtractor()

ipca_hog = None
ipca_lbp = None
if IPCA_HOG_PATH and Path(IPCA_HOG_PATH).exists():
    print(f"[INDEXER] Loading IPCA HOG from {IPCA_HOG_PATH}")
    ipca_hog = joblib.load(IPCA_HOG_PATH)
else:
    if IPCA_HOG_PATH:
        print(f"[INDEXER] Warning: IPCA_HOG path provided but file not found: {IPCA_HOG_PATH}")


def _to_list(vec: np.ndarray):
    return vec.astype(float).tolist()

def _norm(vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return vec.tolist()
    return (vec / (n + 0.0)).astype(np.float32).tolist()

# ---------- Directory walker (numeric subfolders) ----------
def numeric_sorted_dirs(root: Path):
    # yield directories whose name is numeric, sorted numerically
    dirs = [p for p in root.iterdir() if p.is_dir()]
    def key(p):
        try:
            return int(p.name)
        except Exception:
            return float("inf")
    return sorted(dirs, key=key)

def numeric_sorted_files(dir_path: Path, ext: str):
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ext]
    def key(p):
        try:
            return int(p.stem)
        except Exception:
            return float("inf")
    return sorted(files, key=key)

# ---------- Main indexing loop ----------
def index_all():
    actions = []
    total_indexed = 0
    start_time = time.time()

    dirs = numeric_sorted_dirs(DATA_DIR)
    if args.start_dir is not None or args.end_dir is not None:
        filtered = []
        for p in dirs:
            try:
                n = int(p.name)
            except Exception:
                continue
            if args.start_dir is not None and n < args.start_dir:
                continue
            if args.end_dir is not None and n > args.end_dir:
                continue
            filtered.append(p)
        dirs = filtered

    print(f"[INDEXER] Found {len(dirs)} numeric directories under {DATA_DIR}")

    for d in dirs:
        print(f"[INDEXER] Processing directory: {d}")
        files = numeric_sorted_files(d, EXT)
        if not files:
            print(f"[INDEXER] No files with extension {EXT} in {d}, skipping.")
            continue

        # iterate files in sorted order
        for img_path in tqdm(files, desc=f"Dir {d.name}", unit="img"):
            try:
                img_id = f"{d.name}_{img_path.stem}"
                # compute features
                v = vgg.extract(str(img_path))  # 512
                h = hog.extract(str(img_path))
                l = lbp.extract(str(img_path))

                tags_list = []
                tag_file = TAGS_ROOT / d.name / f"{img_path.stem}.txt"
                if tag_file.exists():
                    with open(tag_file, "r", encoding="utf-8") as f:
                        # assuming one tag per line
                        tags_list = [line.strip() for line in f if line.strip()]
                

                if ipca_hog is not None:
                    try:
                        h = ipca_hog.transform(h.reshape(1, -1)).reshape(-1)
                    except Exception as e:
                        # fallback: use raw hog
                        print(f"[WARN] PCA HOG transform failed for {img_path}: {e}")

                if ipca_lbp is not None:
                    try:
                        l = ipca_lbp.transform(l.reshape(1, -1)).reshape(-1)
                    except Exception as e:
                        print(f"[WARN] PCA LBP transform failed for {img_path}: {e}")

                # final L2-normalize (safe)
                v_list = _norm(v)
                h_list = _norm(np.asarray(h, dtype=np.float32))
                l_list = _norm(np.asarray(l, dtype=np.float32))

                action = {
                    "_op_type": OP,
                    "_index": INDEX_NAME,
                    "_id": img_id,
                    "_source": {
                        "image_id": img_id,
                        "url": str(img_path),   # local path; change to public URL if you have one
                        "vgg_vector": v_list,
                        "hog_vector": h_list,
                        "lbp_vector": l_list,
                        "tags": tags_list
                    }
                }
                actions.append(action)

            except Exception as e:
                print(f"[ERROR] processing {img_path}: {e}")

            # flush batch
            if len(actions) >= BATCH:
                try:
                    helpers.bulk(es, actions)
                    total_indexed += len(actions)
                except Exception as e:
                    print(f"[ERROR] bulk indexing failed: {e}")
                actions = []

        # after each directory, flush remaining actions (optional)
        if actions:
            try:
                helpers.bulk(es, actions)
                total_indexed += len(actions)
            except Exception as e:
                print(f"[ERROR] bulk indexing failed (end of dir): {e}")
            actions = []

        # optional progress log per dir
        elapsed = time.time() - start_time
        print(f"[INDEXER] Completed dir {d.name}. Total indexed so far: {total_indexed}. Elapsed: {elapsed:.1f}s")

    # final flush (should be empty)
    if actions:
        try:
            helpers.bulk(es, actions)
            total_indexed += len(actions)
        except Exception as e:
            print(f"[ERROR] final bulk indexing failed: {e}")

    total_time = time.time() - start_time
    print(f"[INDEXER] Done. Total indexed: {total_indexed}. Time: {total_time:.1f}s. Rate: {total_indexed / (total_time + 1e-9):.2f} docs/s")

if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise SystemExit(f"Data dir not found: {DATA_DIR}")
    # optionally check index exists
    if not es.indices.exists(index=INDEX_NAME):
        print(f"[INDEXER] Warning: index {INDEX_NAME} does not exist in ES at {ES_HOST}. Create it first or indexing will create it with default mapping.")
    index_all()