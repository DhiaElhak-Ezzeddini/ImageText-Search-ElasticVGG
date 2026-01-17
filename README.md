<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.12-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Elasticsearch-8.6-005571?style=for-the-badge&logo=elasticsearch&logoColor=white" alt="Elasticsearch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.95-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

<h1 align="center">🔍 Hybrid Image-Text Search Engine</h1>

<p align="center">
  <strong>A powerful Content-Based Image Retrieval (CBIR) system combining deep learning visual features with text-based tag search using Elasticsearch vector similarity.</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-project-structure">Structure</a>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Feature Extraction](#-feature-extraction)
- [Data Structure](#-data-structure)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Notebooks](#-notebooks)
- [License](#-license)

---

## 🎯 Overview

This project implements a **hybrid image-text search engine** that enables users to find visually similar images using three powerful search modes:

| Mode | Description |
|------|-------------|
| 🖼️ **Image Search** | Upload an image to find visually similar images based on deep learning features |
| 📝 **Text Search** | Search by keywords/tags to find images matching your description |
| 🔀 **Hybrid Search** | Combine image similarity with text relevance using adjustable weights |

The system leverages **Elasticsearch's vector similarity search** capabilities with multiple feature extractors (VGG16, HOG, LBP) to provide accurate and flexible image retrieval.

---

## ✨ Features

### Core Capabilities

- **🧠 Multi-Feature Extraction**
  - **VGG16**: 512-dimensional deep learning features (semantic understanding)
  - **HOG**: Histogram of Oriented Gradients (shape & edge detection)
  - **LBP**: Local Binary Patterns (texture analysis)

- **⚡ High-Performance Search**
  - Elasticsearch-based vector similarity using `dotProduct` scoring
  - PCA dimensionality reduction for HOG features (256 components)
  - Bulk indexing with configurable batch sizes

- **🎛️ Flexible Weight Control**
  - Adjust individual feature weights (VGG, HOG, LBP)
  - Balance image vs text relevance in hybrid mode
  - Real-time weight adjustment via web interface

- **🌐 Modern Web Interface**
  - Drag-and-drop image upload
  - Responsive grid layout with Tailwind CSS
  - Real-time search results display
  - Interactive weight sliders

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Docker Compose                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Frontend   │────│   Backend    │────│     Elasticsearch        │  │
│  │  (Embedded)  │    │   (FastAPI)  │    │    (Vector Store)        │  │
│  │   Port 8000  │    │   Port 8000  │    │      Port 9200           │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                              │                        │                  │
│                              │                        │                  │
│  ┌──────────────┐    ┌──────┴───────┐    ┌──────────┴──────────────┐  │
│  │    Kibana    │    │    Trainer   │    │        Indexer          │  │
│  │  (Monitoring)│    │   (PCA)      │    │   (Bulk Indexing)       │  │
│  │   Port 5601  │    │              │    │                         │  │
│  └──────────────┘    └──────────────┘    └─────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Components

| Service | Description | Port |
|---------|-------------|------|
| **elasticsearch** | Vector storage & similarity search engine | 9200 |
| **kibana** | Elasticsearch monitoring & visualization | 5601 |
| **backend** | FastAPI REST API + Web Frontend | 8000 |
| **indexer** | Batch image feature extraction & indexing | - |
| **trainer** | IncrementalPCA model training for HOG | - |

---

## 🛠️ Technology Stack

### Backend & ML
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10 | Runtime environment |
| TensorFlow | 2.12.0 | VGG16 deep feature extraction |
| FastAPI | Latest | REST API framework |
| Uvicorn | Latest | ASGI server |
| scikit-learn | Latest | PCA & machine learning utilities |
| scikit-image | Latest | HOG & LBP extraction |
| OpenCV | Latest | Image processing |
| Pillow | Latest | Image I/O |

### Infrastructure
| Technology | Version | Purpose |
|------------|---------|---------|
| Elasticsearch | 8.6.0 | Vector similarity search |
| Kibana | 8.6.0 | Monitoring & visualization |
| Docker Compose | 3.8 | Container orchestration |

### Frontend
| Technology | Purpose |
|------------|---------|
| Tailwind CSS | Styling & responsive design |
| Vanilla JavaScript | Interactive features |

---

## 📦 Prerequisites

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **8GB+ RAM** recommended (Elasticsearch requires significant memory)
- **GPU** (optional, for faster feature extraction)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ImageTextSearch_Project
```

### 2. Configure Environment

Create a `.env` file in the project root (or modify the existing one):

```env
# Elasticsearch Configuration
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_PORT=9200
INDEX_NAME=images

# Feature Extraction
VGG_DIMS=512
HOG_DIMS=256
LBP_DIMS=26

# Indexer Settings
BATCH_SIZE=256
START_DIR=0
END_DIR=99
```

### 3. Prepare Your Data

Organize your images and tags following this structure:

```
Data/
├── images/
│   ├── 0/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── 1/
│   └── ...
└── tags/
    ├── 0/
    │   ├── 0.txt
    │   ├── 1.txt
    │   └── ...
    ├── 1/
    └── ...
```

### 4. Start the Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Start specific services
docker-compose up -d elasticsearch kibana backend
```

### 5. Train PCA Model (First Time Only)

```bash
docker-compose up trainer
```

### 6. Index Your Images

```bash
docker-compose up indexer
```

### 7. Access the Application

- **Web Interface**: http://localhost:8000
- **Kibana**: http://localhost:5601
- **API Documentation**: http://localhost:8000/docs

---

## 📖 Usage

### Web Interface

1. **Open** http://localhost:8000 in your browser
2. **Choose** a search mode:
   - **Image Only**: Upload an image to find similar images
   - **Text Only**: Enter keywords/tags
   - **Hybrid**: Combine both methods
3. **Adjust weights** using the sliders:
   - VGG Weight: Semantic similarity
   - HOG Weight: Shape/edge similarity
   - LBP Weight: Texture similarity
   - Image/Text Weight: Balance between visual and text search
4. **Upload** an image via drag-and-drop or file selection
5. **Enter** search text (for text/hybrid modes)
6. **Click** "Search" to view results

### Search Examples

#### Image-Only Search
```bash
curl -X POST http://localhost:8000/search \
  -F "mode=image" \
  -F "image=@your_image.jpg" \
  -F "w_vgg=1.0" \
  -F "w_hog=0.5" \
  -F "w_lbp=0.3" \
  -F "top_k=20"
```

#### Text-Only Search
```bash
curl -X POST http://localhost:8000/search \
  -F "mode=text" \
  -F "text=sunset beach ocean" \
  -F "top_k=20"
```

#### Hybrid Search
```bash
curl -X POST http://localhost:8000/search \
  -F "mode=hybrid" \
  -F "image=@your_image.jpg" \
  -F "text=nature landscape" \
  -F "w_vgg=1.0" \
  -F "w_hog=0.5" \
  -F "w_lbp=0.3" \
  -F "w_img=0.7" \
  -F "w_text=0.3" \
  -F "top_k=20"
```

---

## 📚 API Reference

### POST `/search`

Unified search endpoint supporting all search modes.

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | string | Yes | - | Search mode: `image`, `text`, or `hybrid` |
| `image` | file | Conditional | - | Image file (required for `image`/`hybrid` modes) |
| `text` | string | Conditional | - | Search query (required for `text`/`hybrid` modes) |
| `top_k` | int | No | 20 | Number of results to return |
| `w_vgg` | float | No | 1.0 | VGG feature weight (0.0 - 1.0) |
| `w_hog` | float | No | 1.0 | HOG feature weight (0.0 - 1.0) |
| `w_lbp` | float | No | 1.0 | LBP feature weight (0.0 - 1.0) |
| `w_img` | float | No | 0.5 | Image score weight for hybrid mode |
| `w_text` | float | No | 0.5 | Text score weight for hybrid mode |

#### Response

```json
{
  "results": [
    {
      "id": "0/123",
      "path": "/data/0/123.jpg",
      "tags": ["sunset", "beach", "ocean"],
      "score": 0.95
    }
  ],
  "total": 20,
  "mode": "hybrid"
}
```

---

## 🧬 Feature Extraction

### VGG16 Extractor
- **Input**: RGB image (224×224)
- **Output**: 512-dimensional L2-normalized vector
- **Model**: VGG16 with ImageNet weights, GlobalAveragePooling
- **Captures**: High-level semantic features, object recognition

### HOG Extractor
- **Input**: Grayscale image (128×128)
- **Output**: PCA-reduced to 256 dimensions
- **Parameters**: 9 orientations, 8×8 pixels per cell
- **Captures**: Edge orientations, shape information

### LBP Extractor
- **Input**: Grayscale image
- **Output**: 26-bin histogram (uniform LBP)
- **Parameters**: P=24 points, R=3 radius
- **Captures**: Texture patterns, local structure

---

## 📁 Data Structure

### Image Organization

Images are organized in numbered directories (0-99) for efficient batch processing:

```
Data/images/
├── 0/           # Directory 0
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── 1/           # Directory 1
├── ...
└── 99/          # Directory 99
```

### Tag Files

Each image has a corresponding `.txt` file with one tag per line:

```
Data/tags/
├── 0/
│   ├── 0.txt    # Tags for image 0/0.jpg
│   ├── 1.txt
│   └── ...
└── ...
```

**Tag File Format:**
```
sunset
beach
ocean
waves
```

### Elasticsearch Document Schema

```json
{
  "id": "0/123",
  "path": "/data/0/123.jpg",
  "tags": ["sunset", "beach", "ocean"],
  "vgg_vector": [0.023, -0.156, ...],
  "hog_vector": [0.089, 0.234, ...],
  "lbp_vector": [0.12, 0.08, ...]
}
```

---

## ⚙️ Configuration

### Docker Compose Resources

Modify `docker-compose.yml` to adjust resource limits:

```yaml
services:
  elasticsearch:
    environment:
      - "ES_JAVA_OPTS=-Xms4g -Xmx4g"  # Heap size
    
  indexer:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 12G
    
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Indexer Arguments

```bash
# Custom start/end directories
python indexer.py --start-dir 0 --end-dir 50

# Resume from specific directory
python indexer.py --start-dir 25
```

---

## 📂 Project Structure

```
ImageTextSearch_Project/
├── 📄 docker-compose.yml     # Container orchestration
├── 📄 .env                   # Environment variables
├── 📄 README.md              # This file
│
├── 📁 backend/               # FastAPI Backend Service
│   ├── 📄 app.py             # Main application & endpoints
│   ├── 📄 Dockerfile         # Backend container config
│   └── 📄 requirements.txt   # Python dependencies
│
├── 📁 worker/                # Background Workers
│   ├── 📄 indexer.py         # Bulk indexing script
│   ├── 📄 train_pca_run.py   # PCA training script
│   ├── 📄 Dockerfile         # Worker container config
│   └── 📄 requirements.txt   # Worker dependencies
│
├── 📁 shared/                # Shared Utilities
│   ├── 📄 feature_extractor.py  # VGG, HOG, LBP extractors
│   └── 📄 pca_tools.py          # IncrementalPCA utilities
│
├── 📁 models/                # Trained Models
│   └── 📄 ipca_hog.joblib    # Trained PCA model for HOG
│
├── 📁 notebooks/             # Jupyter Notebooks
│   ├── 📄 cbir_vgg.ipynb     # VGG feature experiments
│   ├── 📄 cbir_hog.ipynb     # HOG feature experiments
│   ├── 📄 cbir_LBP.ipynb     # LBP feature experiments
│   ├── 📄 cbir_color.ipynb   # Color histogram experiments
│   └── 📄 cbir_texture.ipynb # Texture feature experiments
│
├── 📁 Data/                  # Dataset
│   ├── 📁 images/            # Image files (organized by directory)
│   ├── 📁 tags/              # Tag files (matching image structure)
│   ├── 📁 thumbnails/        # Image thumbnails
│   └── 📁 license/           # Data licensing information
│
└── 📁 logs/                  # Service Logs
    ├── 📁 backend/
    ├── 📁 indexer/
    └── 📁 trainer/
```

---

## 📓 Notebooks

The `notebooks/` directory contains Jupyter notebooks for exploring different CBIR approaches:

| Notebook | Description |
|----------|-------------|
| `cbir_vgg.ipynb` | VGG16 deep feature extraction experiments |
| `cbir_hog.ipynb` | HOG descriptor analysis and tuning |
| `cbir_LBP.ipynb` | LBP texture feature experiments |
| `cbir_color.ipynb` | Color histogram-based retrieval |
| `cbir_texture.ipynb` | Texture-based feature experiments |

---

## 🔧 Troubleshooting

### Common Issues

**Elasticsearch fails to start:**
```bash
# Increase virtual memory limit (Linux)
sudo sysctl -w vm.max_map_count=262144
```

**Out of memory during indexing:**
- Reduce batch size in indexer configuration
- Increase Docker memory limits

**Slow search performance:**
- Ensure Elasticsearch has sufficient heap memory
- Consider reducing feature dimensions

---

## 📄 License

This project uses the MIRFlickr dataset. Please refer to the `Data/license/` directory for licensing information regarding the dataset.

---

## 🙏 Acknowledgments

- **TensorFlow/Keras** for pre-trained VGG16 model
- **Elasticsearch** for vector similarity search capabilities
- **scikit-image** for HOG and LBP implementations
- **MIRFlickr** for the image dataset

---

<p align="center">
  <strong>Built with ❤️ for Image Retrieval Research</strong>
</p>

<p align="center">
  <sub>If you find this project useful, please consider giving it a ⭐</sub>
</p>
