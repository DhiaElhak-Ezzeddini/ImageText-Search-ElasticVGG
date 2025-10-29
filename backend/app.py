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
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid Image Search</title>
    <!-- Load Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom styles for the app */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Custom Drop Zone Highlight */
        #drop-zone.dragging {
            border-color: #2563eb; /* blue-600 */
            background-color: #dbeafe; /* blue-100 */
        }
        
        /* Visually hide the file input */
        #file-input {
            width: 0.1px;
            height: 0.1px;
            opacity: 0;
            overflow: hidden;
            position: absolute;
            z-index: -1;
        }

        /* Loading Spinner */
        .spinner {
            width: 48px;
            height: 48px;
            border: 4px solid #f3f3f3; /* light grey */
            border-top: 4px solid #2563eb; /* blue-600 */
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">

    <div class="container mx-auto p-4 md:p-8 max-w-5xl">
        <header class="text-center mb-8">
            <h1 class="text-4xl font-bold text-gray-800">Hybrid Search Engine</h1>
            <p class="text-lg text-gray-600 mt-2">Search by image, text, or both.</p>
        </header>

        <!-- Search Form -->
        <form id="search-form" class="bg-white p-6 md:p-8 rounded-xl shadow-lg">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <!-- Left Column: Inputs -->
                <div class_ ="space-y-6">
                    <div>
                        <label for="mode-select" class="block text-sm font-medium text-gray-700 mb-1">Search Mode</label>
                        <select name="mode" id="mode-select" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                            <option value="hybrid" selected>Hybrid (Image + Text)</option>
                            <option value="image">Image Only</option>
                            <option value="text">Text Only</option>
                        </select>
                    </div>

                    <div>
                        <label for="text-input" class="block text-sm font-medium text-gray-700 mb-1">Text Query</label>
                        <input type="text" name="text" id="text-input" placeholder="e.g., cat, sunset, beach" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                    </div>

                    <div>
                        <label for="size-input" class="block text-sm font-medium text-gray-700 mb-1">Number of Results</label>
                        <input type="number" name="size" id="size-input" value="50" min="10" max="200" step="10" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                    </div>
                </div>

                <!-- Right Column: File Drop -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Query Image</label>
                    <div id="drop-zone" class="relative w-full h-full min-h-[220px] border-2 border-dashed border-gray-400 rounded-lg flex flex-col justify-center items-center text-center p-6 cursor-pointer hover:border-blue-500 transition-all">
                        <input type="file" name="file" id="file-input" accept="image/*">
                        
                        <!-- This image preview will show up when a file is selected -->
                        <img id="preview-image" src="" alt="Image preview" class="hidden absolute top-0 left-0 w-full h-full object-cover rounded-lg">
                        
                        <!-- This text will hide when a file is selected -->
                        <div id="drop-zone-text" class="flex flex-col items-center text-gray-600">
                             <svg class="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-4-4V6a4 4 0 014-4h5a4 4 0 014 4v6a4 4 0 01-4 4H7z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 16v1a3 3 0 01-3 3H3a3 3 0 01-3-3V6a3 3 0 013-3h2"></path></svg>
                            <p class="mt-2 font-semibold">Drag & drop an image</p>
                            <p class="text-sm">or click to select</p>
                            <span id="file-name" class="text-sm font-medium text-blue-600 mt-2"></span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Submit Button -->
            <div class="mt-8">
                <button type="submit" id="search-button" class="w-full bg-blue-600 text-white font-bold p-4 rounded-lg text-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400">
                    Search
                </button>
            </div>
        </form>
        
        <!-- Results Section -->
        <div class="mt-12">
            <!-- Loading Spinner -->
            <div id="loading" class="hidden flex justify-center items-center py-12">
                <div class="spinner"></div>
                <p class="ml-4 text-lg text-gray-600">Searching...</p>
            </div>

            <!-- Error Message -->
            <div id="error-message" class="hidden bg-red-100 border border-red-400 text-red-700 p-4 rounded-lg"></div>

            <!-- Results Grid -->
            <div id="results-header" class="hidden">
                 <h2 class="text-2xl font-semibold text-gray-800">Results</h2>
                 <p id="results-count" class="text-gray-600 mt-1"></p>
            </div>
            <div id="results-grid" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 mt-6">
                <!-- Results will be injected here -->
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('search-form');
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const fileNameSpan = document.getElementById('file-name');
        const dropZoneText = document.getElementById('drop-zone-text');
        const previewImage = document.getElementById('preview-image');
        
        const loading = document.getElementById('loading');
        const errorMessage = document.getElementById('error-message');
        const resultsHeader = document.getElementById('results-header');
        const resultsCount = document.getElementById('results-count');
        const resultsGrid = document.getElementById('results-grid');
        const searchButton = document.getElementById('search-button');

        let uploadedFile = null;

        // --- Drag and Drop Handlers ---

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // Highlight drop zone when item is dragged over
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragging'), false);
        });

        // Remove highlight when item leaves
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragging'), false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        }

        // --- Click to Upload Handlers ---
        
        // Trigger file input when drop zone is clicked
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        // Handle file selected from input
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        // --- Common File Handling ---
        
        function handleFile(file) {
            if (file && file.type.startsWith('image/')) {
                uploadedFile = file;
                fileNameSpan.textContent = file.name;
                
                // Show image preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    previewImage.classList.remove('hidden');
                };
                reader.readAsDataURL(file);

                dropZoneText.classList.add('hidden');
            } else {
                uploadedFile = null;
                fileNameSpan.textContent = 'Invalid file. Please select an image.';
                previewImage.classList.add('hidden');
                dropZoneText.classList.remove('hidden');
            }
        }

        // --- Form Submission ---
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Show loading, disable button, hide old results
            loading.classList.remove('hidden');
            errorMessage.classList.add('hidden');
            resultsGrid.innerHTML = '';
            resultsHeader.classList.add('hidden');
            searchButton.disabled = true;
            searchButton.textContent = 'Searching...';
            
            // Build FormData
            const formData = new FormData();
            formData.append('mode', document.getElementById('mode-select').value);
            formData.append('text', document.getElementById('text-input').value);
            
            // Only append file if one is selected
            if (uploadedFile) {
                formData.append('file', uploadedFile);
            }

            const mode = formData.get('mode');
            const size = document.getElementById('size-input').value;
            
            // Build query parameters for the URL
            const queryParams = new URLSearchParams({
                mode: mode,
                size: size
            });

            try {
                // We send 'mode' and 'size' in the URL query string
                // We send 'text' and 'file' in the POST body (FormData)
                const res = await fetch(`/search?${queryParams.toString()}`, {
                    method: 'POST',
                    body: formData 
                    // No 'Content-Type' header; browser sets it for FormData
                });

                if (!res.ok) {
                    const errData = await res.json();
                    throw new Error(errData.error || `Server error: ${res.statusText}`);
                }

                const data = await res.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                displayResults(data);

            } catch (err) {
                displayError(err.message);
                console.error('Search failed:', err);
            } finally {
                // Hide loading, re-enable button
                loading.classList.add('hidden');
                searchButton.disabled = false;
                searchButton.textContent = 'Search';
            }
        });

        function displayError(message) {
            errorMessage.textContent = `Error: ${message}`;
            errorMessage.classList.remove('hidden');
            resultsHeader.classList.add('hidden');
            resultsGrid.innerHTML = '';
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
            
            const fragment = document.createDocumentFragment();
            data.results.forEach(r => {
                const img = document.createElement('img');
                img.src = r.url;
                img.title = r.tags.join(', ') + ` (Score: ${r.score.toFixed(3)})`;
                img.alt = r.tags.join(', ') || 'Search result';
                img.className = 'w-full h-40 object-cover rounded-lg shadow-md hover:shadow-xl transition-shadow duration-300 cursor-pointer';
                // Add a simple lightbox-like click
                img.addEventListener('click', () => window.open(r.url, '_blank'));
                fragment.appendChild(img);
            });
            resultsGrid.appendChild(fragment);
        }
    </script>
</body>
</html>

"""
