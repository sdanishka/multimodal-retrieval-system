# 🔍 Multimodal Retrieval System

Production-ready multimodal search system supporting **Face**, **Image**, and **Text** queries with state-of-the-art embeddings and vector search.

---

## 🎯 Overview

This system enables searching across multiple modalities:

| Query Type | Model | Dimension | Use Case |
|------------|-------|-----------|----------|
| **Face** | InsightFace/ArcFace | 512 | Identity matching |
| **Image** | CLIP ViT-B/32 | 512 | Visual similarity |
| **Text** | CLIP Text Encoder | 512 | Semantic search |

### Key Features

- **Entity-Centric Design**: Not just images, but entities with multiple embeddings
- **Weighted Fusion Ranking**: Combine scores from multiple modalities
- **Real-time Search**: Sub-100ms search across vectors
- **Production Ready**: FastAPI backend, Next.js frontend, Colab pipeline

---

## 🏗️ Architecture

```
QUERY INPUT → EMBEDDING LAYER → FAISS INDEXES → FUSION RANKING → RESULTS

Face Query  → ArcFace (512d)  → Face Index   ─┐
Image Query → CLIP (512d)     → Image Index  ─┼→ Weighted Fusion → Entity Results
Text Query  → CLIP Text(512d) → Text Index   ─┘

Fusion Formula:
final_score = 0.5 × face_sim + 0.3 × image_sim + 0.2 × text_sim
```

---

## 📁 Project Structure

```
multimodal-retrieval/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── embed.py             # Embedding engine
│   ├── search.py            # Multimodal search
│   ├── entity_store.py      # Entity management
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   └── styles/globals.css
├── notebooks/
│   └── multimodal_retrieval_pipeline.ipynb
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

### Option 2: Google Colab (GPU)

1. Open `notebooks/multimodal_retrieval_pipeline.ipynb` in Colab
2. Enable GPU runtime
3. Run all cells
4. Use ngrok URL in frontend

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search/face` | Search by face |
| `POST` | `/search/image` | Search by image |
| `POST` | `/search/text` | Search by text |
| `POST` | `/index/entity` | Index entity |
| `GET` | `/stats` | System stats |

### Example Request

```bash
curl -X POST "http://localhost:8000/search/text" \
  -F "query=man with beard" \
  -F "top_k=10"
```

### Response

```json
{
  "query_type": "text",
  "total_results": 5,
  "results": [{
    "entity_id": "person_042",
    "score": 0.91,
    "images": [...],
    "metadata": {...},
    "match_details": {"face": 0.85, "image": 0.72, "text": 0.95}
  }],
  "search_time_ms": 45.23
}
```

---

## ⚖️ Fusion Weights

| Query Type | Face | Image | Text |
|------------|------|-------|------|
| Face | 0.70 | 0.30 | 0.00 |
| Image | 0.30 | 0.60 | 0.10 |
| Text | 0.10 | 0.40 | 0.50 |

---

## 📊 Performance

- Face embedding: ~50ms
- Image embedding: ~30ms
- Search (1M vectors): ~5ms
- Total latency: <100ms

---

## 📝 License

MIT License
