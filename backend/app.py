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

# ---------------- /search/mixed endpoint (corrected) ----------------
@app.post("/search/mixed")
async def search_mixed(
    file: UploadFile = File(...),
    text: str = Query("", description="Optional text query"),
    w_vgg: float = 0.7,
    w_hog: float = 0.2,
    w_lbp: float = 0.1,
    w_img: float = 0.65,
    w_text: float = 0.35,
    size: int = 50
):
    """Hybrid image + text search (image-priority)"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2️⃣ Extract features
    vgg_q = normalize(vgg_ext.extract(img)).reshape(-1).astype(np.float32)
    hog_q = normalize(hog_ext.extract(img)).reshape(-1).astype(np.float32)
    lbp_q = normalize(lbp_ext.extract(img)).reshape(-1).astype(np.float32)
   

    # Apply PCA if available
    if ipca_hog is not None:
        hog_q = ipca_hog.transform(hog_q.reshape(1, -1)).reshape(-1).astype(np.float32)

    es_query = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}} if not text else {"match": {"tags": text}},
                "script": {
                    "source": "params.w_im *(params.w1 * cosineSimilarity(params.vgg, 'vgg_vector') + params.w2 * cosineSimilarity(params.hog, 'hog_vector') +params.w3 * cosineSimilarity(params.lbp, 'lbp_vector') +3.0) + params.w_text * _score ",
                    "params": {
                        "hog": to_list(hog_q),
                        "vgg": to_list(vgg_q),
                        "lbp": to_list(lbp_q),
                        "w1" : w_vgg,
                        "w2" : w_hog,
                        "w3" : w_lbp,
                        "w_im":w_img,
                        "w_text":w_text,

                    }
                }
            }
        },
        "_source": ["url", "tags"]
    }


    res = es.search(index=ES_INDEX, body=es_query)
    hits = res.get("hits", {}).get("hits", [])

    # 5️⃣ Prepare results
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

    return JSONResponse(content={"results": results})


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
<h2>🔍 Search by Image + Text</h2>
<form id="form">
  <input type="file" name="file" required><br><br>
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
  const res = await fetch('/search/mixed', { method: 'POST', body: formData });
  const data = await res.json();
  resultsDiv.innerHTML = '';
  if (!data.results.length) {
    resultsDiv.innerHTML = '<p>No results found.</p>';
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