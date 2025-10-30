# ImageText-Search — Elastic + VGG based Image & Text Search

**ImageText-Search** is a compact, production-ready pipeline for building multimodal image-text search using visual embeddings extracted with a pre-trained VGG model and a lightweight Elasticsearch index for nearest-neighbor/semantic search.  
The project provides data pipelines, a backend API, and worker components to index images, store metadata, and perform fast similarity / text-based retrieval.

---

## 🚀 Key features

- Visual feature extraction using **VGG16** (transfer learning — image embeddings).
- Indexing of image embeddings + metadata in **Elasticsearch** for efficient nearest-neighbor search.
- Combined image & text search (visual similarity + metadata / captions).
- Simple HTTP **backend** API for querying and indexing.
- Background **worker** for bulk preprocessing and indexing.
- Jupyter **notebooks** for experiments, feature visualization, and demo searches.
- Docker & `docker-compose` for easy local deployment.

---

## 🧭 Repository structure

├─ backend/ # Flask/FastAPI backend (API endpoints, search routes)
├─ worker/ # Background worker for feature extraction & bulk indexing
├─ models/ # Pre-trained model scripts / saved weights / utils
├─ notebooks/ # Exploratory notebooks (demo, evaluation, visualization)
├─ shared/ # Shared utilities and data schemas
├─ docker-compose.yml # Compose file to spin up Elasticsearch + backend + worker
└─ .gitignore



---

## 🔧 Prerequisites

- Docker & Docker Compose (recommended for reproducible setup)
- Python 3.9+ (if you run services locally without Docker)
- (Optional) GPU + CUDA if you want accelerated feature extraction

---

## 🏁 Quick start (Docker)

> This will run Elasticsearch, the backend API and a worker service for indexing.

1. Clone the repo:
```bash
git clone https://github.com/DhiaElhak-Ezzeddini/ImageText-Search-ElasticVGG.git
cd ImageText-Search-ElasticVGG
```

2. Start Services:
```bash
docker-compose up --build
```

3. Visit the backend API (example):
```bash
http://localhost:8000/   # or :5000 depending on backend config
```
## 📌 How it works (high level)

1. Feature extraction: Images are passed through a pre-trained VGG model (typically up to the last pooling layer) to produce compact embeddings (vectors).

2. Indexing: Each image embedding is normalized and stored into an Elasticsearch index together with the image metadata (filename, captions, tags, URL, etc.).

3. Searching:

   - Image search: Upload/query image → compute embedding → nearest neighbor search on Elasticsearch by cosine or L2 distance.

   - Text search: Use Elasticsearch full-text queries on caption/metadata, optionally combine with visual scores for hybrid ranking.

4. Worker: A worker handles CPU/GPU-bound tasks (batch embedding extraction & bulk indexing) so the backend remains responsive.

