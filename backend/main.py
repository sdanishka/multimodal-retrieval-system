"""
Multimodal Retrieval System - FastAPI Backend
Using Danish's Smart Gallery Model Configuration
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from embed import EmbeddingEngine
from search import MultimodalSearchEngine
from entity_store import EntityStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration for your Smart Gallery models
CONFIG = {
    "index_dir": Path("indexes"),
    "entity_store_path": Path("entity_store.json"),
    "face_embedding_dim": 512,    # InsightFace buffalo_l
    "image_embedding_dim": 768,   # CLIP ViT-L-14
    "text_embedding_dim": 768,    # CLIP ViT-L-14 text
    "fusion_weights": {"face": 0.5, "image": 0.3, "text": 0.2},
    "use_dino": False,
    "yolo_model_path": None,  # Set to your model path
}

embedding_engine: Optional[EmbeddingEngine] = None
search_engine: Optional[MultimodalSearchEngine] = None
entity_store: Optional[EntityStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_engine, search_engine, entity_store
    
    logger.info("🚀 Starting Multimodal Retrieval System...")
    
    embedding_engine = EmbeddingEngine(
        use_dino=CONFIG["use_dino"],
        yolo_model_path=CONFIG["yolo_model_path"]
    )
    await embedding_engine.initialize()
    logger.info(f"✅ Models: {embedding_engine.get_loaded_models()}")
    
    entity_store = EntityStore(CONFIG["entity_store_path"])
    entity_store.load()
    logger.info(f"✅ Entities: {len(entity_store)}")
    
    search_engine = MultimodalSearchEngine(
        index_dir=CONFIG["index_dir"],
        entity_store=entity_store,
        embedding_engine=embedding_engine,
        fusion_weights=CONFIG["fusion_weights"]
    )
    await search_engine.initialize()
    logger.info(f"✅ Indexes: {search_engine.get_index_sizes()}")
    
    yield
    
    if search_engine: search_engine.save_indexes()
    if entity_store: entity_store.save()


app = FastAPI(
    title="Multimodal Retrieval System",
    description="Face (512d) + Image (768d) + Text (768d) search using Smart Gallery models",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class SearchResponse(BaseModel):
    query_type: str
    total_results: int
    results: List[Dict[str, Any]]
    search_time_ms: float


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int
    model: str


@app.get("/")
async def root():
    return {"status": "healthy", "service": "Multimodal Retrieval", "models": "Smart Gallery Config"}


@app.get("/stats")
async def get_stats():
    return {
        "total_entities": len(entity_store) if entity_store else 0,
        "index_sizes": search_engine.get_index_sizes() if search_engine else {},
        "models_loaded": embedding_engine.get_loaded_models() if embedding_engine else [],
        "embedding_dims": {"face": 512, "image": 768, "text": 768}
    }


@app.post("/embed/face", response_model=EmbedResponse)
async def embed_face(file: UploadFile = File(...)):
    contents = await file.read()
    embedding = await embedding_engine.embed_face(contents)
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected")
    return EmbedResponse(embedding=embedding.tolist(), dimension=512, model="insightface_buffalo_l")


@app.post("/embed/image", response_model=EmbedResponse)
async def embed_image(file: UploadFile = File(...)):
    contents = await file.read()
    embedding = await embedding_engine.embed_image(contents)
    return EmbedResponse(embedding=embedding.tolist(), dimension=768, model="clip_vit_l_14")


@app.post("/embed/text", response_model=EmbedResponse)
async def embed_text(query: str = Form(...)):
    embedding = await embedding_engine.embed_text(query)
    return EmbedResponse(embedding=embedding.tolist(), dimension=768, model="clip_vit_l_14_text")


@app.post("/search/face", response_model=SearchResponse)
async def search_by_face(file: UploadFile = File(...), top_k: int = Form(default=10)):
    start = time.time()
    contents = await file.read()
    try:
        results = await search_engine.search_by_face(contents, top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return SearchResponse(query_type="face", total_results=len(results), results=results, search_time_ms=round((time.time()-start)*1000, 2))


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(default=10)):
    start = time.time()
    contents = await file.read()
    results = await search_engine.search_by_image(contents, top_k)
    return SearchResponse(query_type="image", total_results=len(results), results=results, search_time_ms=round((time.time()-start)*1000, 2))


@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(query: str = Form(...), top_k: int = Form(default=10)):
    start = time.time()
    results = await search_engine.search_by_text(query, top_k)
    return SearchResponse(query_type="text", total_results=len(results), results=results, search_time_ms=round((time.time()-start)*1000, 2))


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), confidence: float = Form(default=0.25)):
    contents = await file.read()
    detections = await embedding_engine.detect_objects(contents, confidence)
    return {"total_detections": len(detections), "detections": detections}


@app.post("/index/entity")
async def index_entity(
    entity_id: str = Form(...),
    face_image: Optional[UploadFile] = File(default=None),
    general_image: Optional[UploadFile] = File(default=None),
    text_description: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None)
):
    generated = {"face": False, "image": False, "text": False}
    meta = json.loads(metadata) if metadata else {}
    
    face_emb = image_emb = text_emb = None
    
    if face_image:
        face_emb = await embedding_engine.embed_face(await face_image.read())
        generated["face"] = face_emb is not None
    
    if general_image:
        image_emb = await embedding_engine.embed_image(await general_image.read())
        generated["image"] = True
    
    if text_description:
        text_emb = await embedding_engine.embed_text(text_description)
        generated["text"] = True
        meta["description"] = text_description
    
    if not any(generated.values()):
        raise HTTPException(status_code=400, detail="No embeddings generated")
    
    entity = entity_store.create_or_update_entity(
        entity_id=entity_id,
        face_embedding=face_emb,
        image_embedding=image_emb,
        text_embedding=text_emb,
        metadata=meta
    )
    await search_engine.index_entity(entity)
    
    return {"entity_id": entity_id, "success": True, "embeddings_generated": generated}


@app.get("/entities")
async def list_entities(limit: int = 100, offset: int = 0):
    entities = entity_store.list_entities(limit=limit, offset=offset)
    return {"total": len(entity_store), "entities": [e.to_dict() for e in entities]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
