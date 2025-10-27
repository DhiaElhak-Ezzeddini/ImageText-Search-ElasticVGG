# backend/app.py
from fastapi import FastAPI, File, UploadFile, Query # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from elasticsearch import Elasticsearch # type: ignore
import numpy as np
import cv2
import os
import joblib # type: ignore
from pathlib import Path
from shared.feature_extractor import VGGExtractor, HOGExtractor, LBPExtractor  # feature extractors

app = FastAPI(title="Image-Text Hybrid Search API")

# ---------------- Elasticsearch client ----------------
ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX", "images")
es = Elasticsearch(ES_HOST)

# ---------------- Feature extractors ----------------
vgg_ext = VGGExtractor()
hog_ext = HOGExtractor()
lbp_ext = LBPExtractor()

# ---------------- Optional PCA models ----------------
IPCA_HOG_PATH = Path(os.getenv("IPCA_HOG", "/models/ipca_hog.joblib"))
IPCA_LBP_PATH = Path(os.getenv("IPCA_LBP", "/models/ipca_lbp.joblib"))

ipca_hog = joblib.load(IPCA_HOG_PATH) if IPCA_HOG_PATH.exists() else None
ipca_lbp = joblib.load(IPCA_LBP_PATH) if IPCA_LBP_PATH.exists() else None

# ---------------- Helper functions ----------------
def to_list(a):
    return a.tolist() if isinstance(a, np.ndarray) else list(a)

def normalize(vec: np.ndarray):
    n = np.linalg.norm(vec)
    return (vec / (n + 1e-9)).astype(np.float32)

# ---------------- /search/mixed ----------------
@app.post("/search/mixed")
async def search_mixed(
    file: UploadFile = File(...),
    text: str = Query("", description="Optional text query"),
    w_vgg: float = 0.6,
    w_hog: float = 0.3,
    w_lbp: float = 0.1,
    w_img: float = 0.8,
    w_text: float = 0.2,
    size: int = 50
):
    """
    Perform hybrid search:
    - Always prioritize image similarity (VGG/HOG/LBP)
    - Optionally refine with text/tags match
    """
    # read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # compute features
    vgg_q = normalize(vgg_ext.extract(img))
    hog_q = normalize(hog_ext.extract(img))
    lbp_q = normalize(lbp_ext.extract(img))

    # apply PCA if available
    if ipca_hog is not None:
        hog_q = ipca_hog.transform(hog_q.reshape(1, -1)).reshape(-1)
    if ipca_lbp is not None:
        lbp_q = ipca_lbp.transform(lbp_q.reshape(1, -1)).reshape(-1)

    # build query
    body = {
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"tags": text}} if text else {"match_all": {}}
                        ]
                    }
                },
                "script": {
                    "source": """
                        double s1 = 0.0;
                        double s2 = 0.0;
                        double s3 = 0.0;
                        double t = 0.0;
                        if (params.vgg != null && doc['vgg_vector'].size()!=0) {
                            s1 = cosineSimilarity(params.vgg, 'vgg_vector');
                        }
                        if (params.hog != null && doc['hog_vector'].size()!=0) {
                            s2 = cosineSimilarity(params.hog, 'hog_vector');
                        }
                        if (params.lbp != null && doc['lbp_vector'].size()!=0) {
                            s3 = cosineSimilarity(params.lbp, 'lbp_vector');
                        }
                        if (_score != null) {
                            t = _score; // text match contribution
                        }
                        return params.w_img * (params.w1*s1 + params.w2*s2 + params.w3*s3)
                               + params.w_text * t;
                    """,
                    "params": {
                        "vgg": to_list(vgg_q),
                        "hog": to_list(hog_q),
                        "lbp": to_list(lbp_q),
                        "w1": w_vgg,
                        "w2": w_hog,
                        "w3": w_lbp,
                        "w_img": w_img,
                        "w_text": w_text,
                    },
                },
            }
        },
        "_source": ["url", "tags"]
    }

    # run query
    res = es.search(index=ES_INDEX, body=body)
    hits = res["hits"]["hits"]

    # return URLs and metadata
    results = []
    for h in hits:
        src = h["_source"]
        results.append({
            "id": h["_id"],
            "score": h["_score"],
            "url": src.get("url"),
            "tags": src.get("tags", [])
        })

    return JSONResponse(content={"results": results})


# ---------------- /search/multi-stage ----------------
@app.post("/search/multi-stage")
async def search_multistage(
    file: UploadFile = File(...),
    w_vgg: float = 0.6,
    w_hog: float = 0.3,
    w_lbp: float = 0.1,
    per_field_k: int = 200,
    final_k: int = 10
):
    """Two-stage retrieval: per-field kNN then re-ranking"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    vgg_q = normalize(vgg_ext.extract(img))
    hog_q = normalize(hog_ext.extract(img))
    lbp_q = normalize(lbp_ext.extract(img))

    candidates = {}
    for field, q in [("vgg_vector", vgg_q), ("hog_vector", hog_q), ("lbp_vector", lbp_q)]:
        body = {
            "size": per_field_k,
            "query": {"knn": {field: {"vector": to_list(q), "k": per_field_k}}},
            "_source": ["url", "tags"]
        }
        try:
            res = es.search(index=ES_INDEX, body=body)
        except Exception:
            res = {"hits": {"hits": []}}
        for h in res["hits"]["hits"]:
            candidates[h["_id"]] = True

    if not candidates:
        return {"results": []}

    # fetch candidate docs and re-score manually
    mget = es.mget(index=ES_INDEX, body={"ids": list(candidates.keys())})
    scored = []
    for doc in mget["docs"]:
        if not doc.get("found"):
            continue
        src = doc["_source"]
        def dot(a, b): return float(np.dot(np.array(a, dtype=np.float32), b))
        s_vgg = dot(src["vgg_vector"], vgg_q)
        s_hog = dot(src["hog_vector"], hog_q)
        s_lbp = dot(src["lbp_vector"], lbp_q)
        score = w_vgg * s_vgg + w_hog * s_hog + w_lbp * s_lbp
        scored.append({
            "id": doc["_id"],
            "score": score,
            "url": src["url"],
            "tags": src.get("tags", [])
        })

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:final_k]
    return {"results": scored}
