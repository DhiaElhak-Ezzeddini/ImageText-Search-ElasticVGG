from fastapi import FastAPI, File, UploadFile, Query, Form # type: ignore
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

def build_keyword_query(text: str):
    """
    Build a safe keyword-based query for exact tag matches.
    Splits the input text into tokens (by space/comma) and matches any of them.
    """
    if not text:
        return {"match_all": {}}

    # This assumes your 'tags' field is a keyword array and you indexed them in lowercase.
    # If not, a 'term' query is case-sensitive and will fail.
    tokens = [t.strip().lower() for t in text.replace(",", " ").split() if t.strip()]
    
    # If your tags are not lowercased at index time, this 'term' query will
    # fail for case mismatches (e.g., searching "cat" for indexed "Cat").
    # The mapping you provided doesn't show a 'normalizer', so this is a
    # potential issue to be aware of.
    return {
        "bool": {
            "should": [{"term": {"tags": t}} for t in tokens],
            "minimum_should_match": 1
        }
    }

# ---------------- Unified Search Endpoint ----------------
@app.post("/search")
async def search(
    file: UploadFile | None = File(None),
    # *** FIX: Changed Query to Form to read 'text' from the multipart/form-data body ***
    text: str = Form("", description="Optional text query"),
    mode: str = Query("hybrid", enum=["image", "text", "hybrid"]),
    w_vgg: float = 0.8,
    w_hog: float = 0.1,
    w_lbp: float = 0.1,
    w_img: float = 0.95,
    w_text: float = 0.05,
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
    if file and file.filename and mode in ["image", "hybrid"]:
        contents = await file.read()
        if contents:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # Extract features
                vgg_q = normalize(vgg_ext.extract(img)).reshape(-1).astype(np.float32)
                hog_q = normalize(hog_ext.extract(img)).reshape(-1).astype(np.float32)
                lbp_q = normalize(lbp_ext.extract(img)).reshape(-1).astype(np.float32)

                # Apply PCA if available
                if ipca_hog is not None:
                    hog_q = ipca_hog.transform(hog_q.reshape(1, -1)).reshape(-1).astype(np.float32)
            else:
                # Handle invalid image file
                pass
        else:
            # Handle empty file
            file = None

    # --- 2️⃣ Build Elasticsearch Query ---
    if mode == "text":
        if not text:
             return JSONResponse(content={"mode": mode, "results": [], "error": "Text query cannot be empty in 'text' mode."})
        es_query = {
            "size": size,
            "query": build_keyword_query(text),
            "_source": ["url", "tags"]
        }

    elif mode == "image":
        if vgg_q is None: # Check if image processing failed or no file was sent
             return JSONResponse(content={"mode": mode, "results": [], "error": "Image file is required in 'image' mode."})
        
        # Image-only search
        es_query = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": (
                            "params.w1 * dotProduct(params.vgg, 'vgg_vector') + "
                            "params.w2 * dotProduct(params.hog, 'hog_vector') + "
                            "params.w3 * dotProduct(params.lbp, 'lbp_vector') + 3.0"
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
        if vgg_q is None: # Hybrid mode requires an image
             return JSONResponse(content={"mode": mode, "results": [], "error": "Image file is required in 'hybrid' mode."})
            
        text_query = build_keyword_query(text) # This will be {"match_all": {}} if text is empty

        es_query = {
            "size": size,
            "query": {
                "script_score": {
                    "query": text_query,
                    "script": {
                        "source": (
                            "params.w_im * (params.w1 * dotProduct(params.vgg, 'vgg_vector') + "
                            "params.w2 * dotProduct(params.hog, 'hog_vector') + "
                            "params.w3 * dotProduct(params.lbp, 'lbp_vector') + 3.0) "
                            # Use _score from the text_query. If text_query was match_all, _score is 1.0
                            # If text_query matched, _score is the BM25 score.
                            #"+ params.w_text * _score" 
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
    try:
        res = es.search(index=ES_INDEX, body=es_query)
    except Exception as e:
        return JSONResponse(content={"mode": mode, "results": [], "error": f"Search query failed: {e}"}, status_code=500)

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
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Hybrid Image Search</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    #drop-zone.dragging { border-color: #60a5fa; background-color: rgba(99,102,241,0.06); }
    .spinner { width:48px;height:48px;border:4px solid rgba(255,255,255,0.25);border-top:4px solid #60a5fa;border-radius:50%;animation:spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .glass { background: rgba(255,255,255,0.06); backdrop-filter: blur(10px); border:1px solid rgba(255,255,255,0.06); }
    .slider { -webkit-appearance:none; appearance:none; height:8px; border-radius:9999px; background:linear-gradient(90deg,#60a5fa,#3b82f6); outline:none; }
    .slider::-webkit-slider-thumb { -webkit-appearance:none; width:16px;height:16px;border-radius:9999px;background:white;box-shadow:0 1px 3px rgba(0,0,0,0.3); cursor:pointer; }
  </style>
</head>
<body class="min-h-screen flex flex-col items-center py-8 text-white">
  <header class="text-center mb-6">
    <h1 class="text-4xl font-extrabold">Hybrid Image Search Engine</h1>
    <p class="text-blue-200 mt-1">Combine image and text signals — tune feature and fusion weights.</p>
  </header>

  <main class="w-full max-w-5xl px-4">
    <div class="glass rounded-2xl p-6 shadow-2xl">
      <form id="search-form" class="space-y-6">
        <div class="grid md:grid-cols-2 gap-6">
          <!-- Left: controls -->
          <div class="space-y-4">
            <div>
              <label for="mode-select" class="block text-sm font-medium">Search Mode</label>
              <select id="mode-select" name="mode" class="mt-1 w-full p-3 rounded-lg text-gray-900">
                <option value="hybrid" selected>Hybrid (Image + Text)</option>
                <option value="image">Image Only</option>
                <option value="text">Text Only</option>
              </select>
            </div>

            <div>
              <label for="text-input" class="block text-sm font-medium">Text Query</label>
              <input id="text-input" name="text" type="text" placeholder="e.g. cat, sunset, beach" class="mt-1 w-full p-3 rounded-lg text-gray-900" />
            </div>

            <div>
              <label for="size-input" class="block text-sm font-medium">Number of Results</label>
              <input id="size-input" name="size" type="number" value="50" min="10" max="200" step="10" class="mt-1 w-full p-3 rounded-lg text-gray-900" />
            </div>

            <!-- Fusion weights -->
            <div class="mt-4 p-4 rounded-lg bg-white/5">
              <h3 class="font-medium mb-2">Fusion weights</h3>
              <div class="flex items-center justify-between text-sm mb-2">
                <label>Image vs Text</label><span id="w-im-text-val">95% / 5%</span>
              </div>
              <input id="w-im" type="range" min="0" max="1" step="0.01" value="0.95" class="slider w-full" />
              <p class="text-xs text-blue-200 mt-2">Higher = more importance to the image score when in hybrid mode.</p>
            </div>

          </div>

          <!-- Right: drop zone + feature weights -->
          <div class="space-y-4">
            <label class="block text-sm font-medium">Query Image</label>
            <div id="drop-zone" class="relative h-56 border-2 border-dashed rounded-xl flex items-center justify-center overflow-hidden cursor-pointer">
              <input id="file-input" name="file" type="file" accept="image/*" class="absolute inset-0 opacity-0 cursor-pointer"/>
              <img id="preview-image" src="" alt="Preview" class="hidden absolute inset-0 w-full h-full object-cover" />
              <div id="drop-zone-text" class="text-center px-4">
                <svg class="mx-auto mb-2 w-10 h-10 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-4-4V6a4 4 0 014-4h5a4 4 0 014 4v6a4 4 0 01-4 4H7zM14 16v1a3 3 0 01-3 3H3a3 3 0 01-3-3V6a3 3 0 013-3h2"/></svg>
                <p class="font-semibold">Drag & drop an image or click to select</p>
                <p id="file-name" class="mt-1 text-sm text-blue-200"></p>
              </div>
            </div>

            <div class="mt-2 p-4 rounded-lg bg-white/5">
              <h3 class="font-medium mb-2">Per-feature weights (VGG / HOG / LBP)</h3>
              <div class="flex items-center justify-between text-sm mb-1">
                <label>VGG</label><span id="w-vgg-val">0.80</span>
              </div>
              <input id="w-vgg" type="range" min="0" max="1" step="0.01" value="0.80" class="slider w-full mb-3" />

              <div class="flex items-center justify-between text-sm mb-1">
                <label>HOG</label><span id="w-hog-val">0.10</span>
              </div>
              <input id="w-hog" type="range" min="0" max="1" step="0.01" value="0.10" class="slider w-full mb-3" />

              <div class="flex items-center justify-between text-sm mb-1">
                <label>LBP</label><span id="w-lbp-val">0.10</span>
              </div>
              <input id="w-lbp" type="range" min="0" max="1" step="0.01" value="0.10" class="slider w-full" />
            </div>

          </div>
        </div>

        <div>
          <button id="search-button" type="submit" class="w-full py-3 rounded-lg bg-gradient-to-r from-blue-500 to-indigo-600 font-semibold shadow-md">🔍 Search</button>
        </div>
      </form>
    </div>

    <!-- Loading / Errors -->
    <div id="loading" class="hidden mt-8 flex items-center justify-center">
      <div class="spinner"></div><span class="ml-4">Searching...</span>
    </div>
    <div id="error-message" class="hidden mt-4 bg-red-100 text-red-700 p-3 rounded-lg"></div>

    <!-- Results -->
    <section id="results-section" class="mt-8">
      <div id="results-header" class="hidden text-white mb-4 text-center">
        <h2 class="text-2xl font-semibold">Results</h2>
        <p id="results-count" class="text-blue-200 mt-1"></p>
      </div>
      <div id="results-grid" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4"></div>
    </section>
  </main>

<script>
  // Elements
  const form = document.getElementById('search-form');
  const modeSelect = document.getElementById('mode-select');
  const textInput = document.getElementById('text-input');
  const sizeInput = document.getElementById('size-input');
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const dropZoneText = document.getElementById('drop-zone-text');
  const previewImage = document.getElementById('preview-image');
  const fileNameSpan = document.getElementById('file-name');
  const loading = document.getElementById('loading');
  const errorMessage = document.getElementById('error-message');
  const resultsHeader = document.getElementById('results-header');
  const resultsCount = document.getElementById('results-count');
  const resultsGrid = document.getElementById('results-grid');
  const searchButton = document.getElementById('search-button');

  const wIm = document.getElementById('w-im');
  const wImTextVal = document.getElementById('w-im-text-val');
  const wVgg = document.getElementById('w-vgg');
  const wHog = document.getElementById('w-hog');
  const wLbp = document.getElementById('w-lbp');
  const wVggVal = document.getElementById('w-vgg-val');
  const wHogVal = document.getElementById('w-hog-val');
  const wLbpVal = document.getElementById('w-lbp-val');

  let uploadedFile = null;

  // Update slider labels
  function refreshSliderLabels() {
    const imPct = Math.round(parseFloat(wIm.value) * 100);
    const txtPct = 100 - imPct;
    wImTextVal.textContent = `${imPct}% / ${txtPct}%`;
    wVggVal.textContent = parseFloat(wVgg.value).toFixed(2);
    wHogVal.textContent = parseFloat(wHog.value).toFixed(2);
    wLbpVal.textContent = parseFloat(wLbp.value).toFixed(2);
  }
  [wIm, wVgg, wHog, wLbp].forEach(el=>el.addEventListener('input', refreshSliderLabels));
  refreshSliderLabels();

  // Drag/drop behaviors
  ['dragenter','dragover','dragleave','drop'].forEach(ev=>{
    dropZone.addEventListener(ev, e=>{ e.preventDefault(); e.stopPropagation(); });
  });
  ['dragenter','dragover'].forEach(ev=>dropZone.addEventListener(ev, ()=>dropZone.classList.add('dragging')));
  ['dragleave','drop'].forEach(ev=>dropZone.addEventListener(ev, ()=>dropZone.classList.remove('dragging')));
  dropZone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  // Click to open file selector
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });

  function handleFile(file) {
    if (file && file.type.startsWith('image/')) {
      uploadedFile = file;
      fileNameSpan.textContent = file.name;
      const reader = new FileReader();
      reader.onload = ev => {
        previewImage.src = ev.target.result;
        previewImage.classList.remove('hidden');
        dropZoneText.classList.add('hidden');
      };
      reader.readAsDataURL(file);
    } else {
      uploadedFile = null;
      fileNameSpan.textContent = 'Invalid file (choose an image)';
      previewImage.classList.add('hidden');
      dropZoneText.classList.remove('hidden');
    }
  }

  // Mode logic: enable/disable inputs depending on mode
  function updateModeState() {
    const mode = modeSelect.value;
    if (mode === 'text') {
      // text-only -> disable file area visually
      dropZone.classList.add('opacity-50');
    } else {
      dropZone.classList.remove('opacity-50');
    }
  }
  modeSelect.addEventListener('change', updateModeState);
  updateModeState();

  // Submit handler: builds query params including weight sliders
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    // Validation: hybrid or image requires an image
    const mode = modeSelect.value;
    if ((mode === 'image' || mode === 'hybrid') && !uploadedFile) {
      displayError('Image required for selected mode. Please upload or drop an image.');
      return;
    }
    // Text-only requires text
    if (mode === 'text' && !textInput.value.trim()) {
      displayError("Text query can't be empty in 'text' mode.");
      return;
    }

    // UI state
    loading.classList.remove('hidden');
    errorMessage.classList.add('hidden');
    resultsGrid.innerHTML = '';
    resultsHeader.classList.add('hidden');
    searchButton.disabled = true;
    searchButton.textContent = 'Searching...';

    // Build form data for file + text
    const formData = new FormData();
    formData.append('text', textInput.value || '');
    if (uploadedFile) formData.append('file', uploadedFile);

    // Build query params (FastAPI reads these as Query)
    const params = new URLSearchParams({
      mode,
      size: sizeInput.value || '50',
      w_vgg: wVgg.value,
      w_hog: wHog.value,
      w_lbp: wLbp.value,
      w_img: wIm.value,
      w_text: (1 - parseFloat(wIm.value)).toFixed(2)
    });

    try {
      const res = await fetch(`/search?${params.toString()}`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || res.statusText || 'Server error');
      }
      displayResults(data);
    } catch (err) {
      displayError(err.message);
      console.error('Search failed', err);
    } finally {
      loading.classList.add('hidden');
      searchButton.disabled = false;
      searchButton.textContent = '🔍 Search';
    }
  });

  function displayError(msg) {
    errorMessage.textContent = `Error: ${msg}`;
    errorMessage.classList.remove('hidden');
  }

  function displayResults(data) {
    resultsGrid.innerHTML = '';
    errorMessage.classList.add('hidden');
    if (!data.results || data.results.length === 0) {
      resultsHeader.classList.remove('hidden');
      resultsCount.textContent = 'No results found.';
      return;
    }
    resultsHeader.classList.remove('hidden');
    resultsCount.textContent = `Found ${data.results.length} results.`;
    data.results.forEach(r => {
      const card = document.createElement('div');
      card.className = 'group relative rounded-lg overflow-hidden shadow-md hover:scale-105 transition-transform duration-200 cursor-pointer';
      const img = document.createElement('img');
      img.src = r.url;
      img.alt = r.tags ? r.tags.join(', ') : 'result';
      img.className = 'w-full h-44 object-cover';
      const overlay = document.createElement('div');
      overlay.className = 'absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center';
      const txt = document.createElement('p');
      txt.className = 'text-white text-sm px-2 text-center';
      txt.textContent = `${r.tags ? r.tags.join(', ') : 'No tags'} — Score: ${parseFloat(r.score).toFixed(3)}`;
      overlay.appendChild(txt);
      card.appendChild(img);
      card.appendChild(overlay);
      card.addEventListener('click', () => window.open(r.url, '_blank'));
      resultsGrid.appendChild(card);
    });
  }
</script>
</body>
</html>
"""
