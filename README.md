# 🔍 Multimodal Retrieval System

A production-ready multimodal search system that enables searching through images using **face recognition**, **visual similarity**, and **natural language text queries**. Built with state-of-the-art deep learning models and optimized for real-time performance.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <img src="docs/demo.gif" alt="Demo" width="800"/>
</p>

## ✨ Features

- **🧑 Face Search** - Find images by uploading a face photo (InsightFace buffalo_l)
- **🖼️ Image Search** - Find visually similar images (CLIP ViT-L-14)
- **📝 Text Search** - Search using natural language descriptions
- **⚡ Real-time Results** - Sub-100ms search latency with FAISS indexing
- **🎯 Weighted Fusion** - Combine multiple modalities for better results
- **🌐 Web Interface** - Modern, responsive Next.js frontend

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js       │────▶│   FastAPI       │────▶│   FAISS         │
│   Frontend      │     │   Backend       │     │   Indexes       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ML Models         │
                    │  • CLIP ViT-L-14    │
                    │  • InsightFace      │
                    │  • YOLOv8x          │
                    └─────────────────────┘
```

## 🤖 Models Used

| Component | Model | Embedding Dim | Purpose |
|-----------|-------|---------------|---------|
| **Face** | InsightFace buffalo_l (ArcFace) | 512 | Face recognition & verification |
| **Image** | CLIP ViT-L-14 | 768 | Visual similarity & semantic understanding |
| **Text** | CLIP ViT-L-14 | 768 | Natural language to image matching |
| **Detection** | YOLOv8x (fine-tuned) | - | Object detection (optional) |

## 📁 Project Structure

```
multimodal-retrieval-system/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── embed.py             # Embedding engine (CLIP, InsightFace)
│   ├── search.py            # FAISS search engine
│   ├── entity_store.py      # Entity management
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Main search page
│   │   └── layout.tsx       # App layout
│   ├── styles/
│   │   └── globals.css      # Tailwind styles
│   ├── .env.local           # API URL configuration
│   └── package.json         # Node dependencies
├── notebooks/
│   └── multimodal_retrieval_pipeline.ipynb  # Colab notebook
└── README.md
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab:
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/multimodal-retrieval-system/blob/main/notebooks/multimodal_retrieval_pipeline.ipynb)

2. Run all cells to start the backend server

3. Copy the ngrok URL and update `frontend/.env.local`

4. Start the frontend locally

### Option 2: Local Development

#### Prerequisites

- Python 3.8+
- Node.js 18+
- CUDA-capable GPU (recommended)

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure API URL
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

Visit `http://localhost:3000` to access the application.

## 📡 API Endpoints

### Search Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search/face` | Search by face image |
| `POST` | `/search/image` | Search by image similarity |
| `POST` | `/search/text` | Search by text description |

### Embedding Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/embed/face` | Generate face embedding (512-dim) |
| `POST` | `/embed/image` | Generate image embedding (768-dim) |
| `POST` | `/embed/text` | Generate text embedding (768-dim) |

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/stats` | System statistics |
| `GET` | `/image/{idx}` | Serve image by index |

### Example Usage

```python
import requests

# Text search
response = requests.post(
    "http://localhost:8000/search/text",
    data={"query": "person smiling outdoors", "top_k": 10}
)
results = response.json()

# Image search
with open("query.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/search/image",
        files={"file": f},
        data={"top_k": 10}
    )
results = response.json()
```

## 🖼️ Adding Your Own Images

### Method 1: Via API

```python
import requests

# Index a new image
with open("my_photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/index/entity",
        files={"general_image": f},
        data={
            "entity_id": "photo_001",
            "text_description": "Beach vacation sunset"
        }
    )
```

### Method 2: Batch Processing (Colab)

```python
# In the Colab notebook
MY_IMAGES_FOLDER = "/content/drive/MyDrive/my_photos"

# Run the indexing code
for img_path in image_files:
    with open(img_path, 'rb') as f:
        emb = embed_image(f.read())
    image_index.add(emb.reshape(1, -1))
    image_paths_list.append(img_path)

print(f"Indexed {len(image_files)} images!")
```

## ⚙️ Configuration

### Fusion Weights

Adjust how different modalities contribute to search results:

```python
# In backend/main.py or via API
fusion_weights = {
    "face": 0.5,   # Weight for face similarity
    "image": 0.3,  # Weight for visual similarity  
    "text": 0.2    # Weight for text/semantic match
}
```

### Detection Thresholds

```python
# Face detection settings
FACE_CONFIG = {
    'det_thresh': 0.4,      # Detection confidence threshold
    'min_face_size': 30,    # Minimum face size in pixels
    'det_size': (640, 640)  # Detection input size
}
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Face Search Latency | ~50ms |
| Image Search Latency | ~30ms |
| Text Search Latency | ~20ms |
| Index Size (10k images) | ~30MB |
| Face Embedding Time | ~15ms/face |
| Image Embedding Time | ~25ms/image |

*Benchmarked on NVIDIA T4 GPU*

## 🛠️ Tech Stack

**Backend:**
- FastAPI - High-performance async API
- PyTorch - Deep learning framework
- FAISS - Vector similarity search
- InsightFace - Face recognition
- OpenCLIP - Image-text embeddings
- Ultralytics - Object detection

**Frontend:**
- Next.js 14 - React framework
- Tailwind CSS - Styling
- Framer Motion - Animations
- Axios - HTTP client

## 🔮 Future Improvements

- [ ] Add video search support
- [ ] Implement face clustering
- [ ] Add image captioning
- [ ] Support multiple face search
- [ ] Add authentication
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Mobile app

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) - Image-text model
- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8

## 📧 Contact

**Danish Afridi** - MSc AI & Data Science, University of Hull

- GitHub: [@DanishAfridi](https://github.com/DanishAfridi)
- LinkedIn: [Danish Afridi](https://linkedin.com/in/danishafridi)

---

<p align="center">
  Made with ❤️ using AI & Deep Learning
</p>
