from fastapi import FastAPI, File, UploadFile, Query # type: ignore
from fastapi.responses import JSONResponse, HTMLResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from elasticsearch import Elasticsearch # type: ignore
import numpy as np
import cv2
import os
import joblib # type: ignore
from pathlib import Path
from shared.feature_extractor import VGGExtractor, HOGExtractor, LBPExtractor

app = FastAPI(title="Image-Text Hybrid Search API")

# Serve static /data
app.mount("/data", StaticFiles(directory="/data"), name="data")

# Elasticsearch client
ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
ES_INDEX = os.getenv("ES_INDEX", "images")
es = Elasticsearch(ES_HOST)

# Feature extractors
vgg_ext = VGGExtractor()
hog_ext = HOGExtractor()
lbp_ext = LBPExtractor()

# Optional PCA models
IPCA_HOG_PATH = Path(os.getenv("IPCA_HOG", "/models/ipca_hog.joblib"))
ipca_hog = joblib.load(IPCA_HOG_PATH) if IPCA_HOG_PATH.exists() else None

# Helper functions
def to_list(a):
    return a.tolist() if isinstance(a, np.ndarray) else list(a)

def normalize(vec: np.ndarray):
    n = np.linalg.norm(vec)
    return (vec / (n + 1e-9)).astype(np.float32)
def build_text_query(text: str):
    """
    Build a robust text query for 'tags'.
    - Prefer match_phrase (good for multi-word tags).
    - Also include a terms query against tags.keyword for exact tag matches.
    """
    if not text:
        return {"match_all": {}}

    # match_phrase for phrases / partial matches
    match_phrase = {"match_phrase": {"tags": {"query": text, "slop": 1}}}

    # terms/term for exact tag tokens (split by whitespace, commas)
    tokens = [t.strip() for t in text.replace(',', ' ').split() if t.strip()]
    if tokens:
        terms = {"terms": {"tags.keyword" if "tags.keyword" else "tags": tokens}}
    else:
        terms = {}

    # Use bool should so either phrase or exact token match will match
    should_clauses = [match_phrase]
    if terms:
        should_clauses.append(terms)

    return {"bool": {"should": should_clauses, "minimum_should_match": 1}}
# ---------------- Unified Search Endpoint ----------------
@app.post("/search")
async def search(
    file: UploadFile | None = File(None),
    text: str = Query("", description="Optional text query"),
    mode: str = Query("hybrid", enum=["image", "text", "hybrid"]),
    w_vgg: float = 0.7,
    w_hog: float = 0.2,
    w_lbp: float = 0.1,
    w_img: float = 0.65,
    w_text: float = 0.35,
    size: int = 50
):
    """
    Unified search endpoint:
    - mode='image'  → search by image only
    - mode='text'   → search by text only
    - mode='hybrid' → search by both (image priority)
    """

    # Initialize variables
    vgg_q = hog_q = lbp_q = None

    # --- 1️⃣ Handle image input if provided ---
    if file and mode in ["image", "hybrid"]:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Extract features
        vgg_q = normalize(vgg_ext.extract(img)).reshape(-1).astype(np.float32)
        hog_q = normalize(hog_ext.extract(img)).reshape(-1).astype(np.float32)
        lbp_q = normalize(lbp_ext.extract(img)).reshape(-1).astype(np.float32)

        # Apply PCA if available
        if ipca_hog is not None:
            hog_q = ipca_hog.transform(hog_q.reshape(1, -1)).reshape(-1).astype(np.float32)

    # --- 2️⃣ Build Elasticsearch Query ---
    if mode == "text":
        # Text-only search
        es_query = {
            "size": size,
            "query": build_text_query(text),
            "_source": ["url", "tags"]
        }

    elif mode == "image":
        # Image-only search
        es_query = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": (
                            "params.w1 * cosineSimilarity(params.vgg, 'vgg_vector') + "
                            "params.w2 * cosineSimilarity(params.hog, 'hog_vector') + "
                            "params.w3 * cosineSimilarity(params.lbp, 'lbp_vector') + 3.0"
                        ),
                        "params": {
                            "hog": to_list(hog_q),
                            "vgg": to_list(vgg_q),
                            "lbp": to_list(lbp_q),
                            "w1": w_vgg,
                            "w2": w_hog,
                            "w3": w_lbp
                        }
                    }
                }
            },
            "_source": ["url", "tags"]
        }

    else:  # hybrid
        es_query = {
            "size": size,
            "query": {
                "script_score": {
                    "query": build_text_query(text),
                    "script": {
                        "source": (
                            "params.w_im * (params.w1 * cosineSimilarity(params.vgg, 'vgg_vector') + "
                            "params.w2 * cosineSimilarity(params.hog, 'hog_vector') + "
                            "params.w3 * cosineSimilarity(params.lbp, 'lbp_vector') + 3.0) "
                            "+ params.w_text * _score"
                        ),
                        "params": {
                            "hog": to_list(hog_q),
                            "vgg": to_list(vgg_q),
                            "lbp": to_list(lbp_q),
                            "w1": w_vgg,
                            "w2": w_hog,
                            "w3": w_lbp,
                            "w_im": w_img,
                            "w_text": w_text,
                        }
                    }
                }
            },
            "_source": ["url", "tags"]
        }

    # --- 3️⃣ Execute search ---
    res = es.search(index=ES_INDEX, body=es_query)
    hits = res.get("hits", {}).get("hits", [])

    # --- 4️⃣ Format results ---
    results = []
    for h in hits:
        src = h["_source"]
        url = src.get("url", "")
        if url and not url.startswith("/data/"):
            url = "/data/" + url.split("/data/")[-1]
        results.append({
            "id": h["_id"],
            "score": h["_score"],
            "url": url,
            "tags": src.get("tags", [])
        })

    return JSONResponse(content={"mode": mode, "results": results})


# ---------------- Simple HTML frontend ----------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Hybrid Image/Text Search</title>
<style>
body { font-family: Arial, sans-serif; margin: 30px; }
img { max-width: 200px; margin: 10px; border: 1px solid #ccc; border-radius: 5px; }
</style>
</head>
<body>
<h2>🔍 Search by Image and/or Text</h2>
<form id="form">
  <label>Search mode:</label><br>
  <select name="mode">
    <option value="hybrid">Hybrid (image priority)</option>
    <option value="image">Image only</option>
    <option value="text">Text only</option>
  </select><br><br>
  
  <input type="file" name="file"><br><br>
  <input type="text" name="text" placeholder="Optional text query"><br><br>
  <button type="submit">Search</button>
</form>
<hr>
<div id="results"></div>

<script>
const form = document.getElementById('form');
const resultsDiv = document.getElementById('results');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  resultsDiv.innerHTML = '<p>Searching...</p>';
  const formData = new FormData(form);
  const res = await fetch('/search?mode=' + formData.get('mode'), { method: 'POST', body: formData });
  const data = await res.json();
  resultsDiv.innerHTML = '<h3>Mode: ' + data.mode + '</h3>';
  if (!data.results.length) {
    resultsDiv.innerHTML += '<p>No results found.</p>';
    return;
  }
  data.results.forEach(r => {
    const img = document.createElement('img');
    img.src = r.url;
    img.title = r.tags.join(', ') + ' (score=' + r.score.toFixed(3) + ')';
    resultsDiv.appendChild(img);
  });
});
</script>
</body>
</html>
"""
