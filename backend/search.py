"""
Multimodal Search Engine - Using Danish's Smart Gallery Config
==============================================================
FAISS index management and multimodal search with weighted fusion.

Embedding Dimensions (matching your ml_pipeline_outputs):
- Face: 512-dim (InsightFace buffalo_l)
- Image CLIP: 768-dim (CLIP ViT-L-14)
- Image DINOv2: 1024-dim (optional)
- Text: 768-dim (CLIP ViT-L-14 text encoder)

Search Flow:
- Face → Entity → All Related Images
- Text → Images + Faces (semantic search)
- Image → Similar Scenes + Extracted Faces
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss

from entity_store import EntityStore, Entity
from embed import EmbeddingEngine

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


class FAISSIndex:
    """
    FAISS index wrapper with entity ID mapping.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
            index_type: 'flat' for exact search, 'ivf' for approximate
        """
        self.dimension = dimension
        self.index_type = index_type
        
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.id_mapping: List[str] = []
        self.entity_positions: Dict[str, List[int]] = {}
    
    def add(self, entity_id: str, embedding: np.ndarray):
        """Add embedding to index."""
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        embedding = embedding.reshape(1, -1)
        
        position = len(self.id_mapping)
        self.index.add(embedding)
        
        self.id_mapping.append(entity_id)
        if entity_id not in self.entity_positions:
            self.entity_positions[entity_id] = []
        self.entity_positions[entity_id].append(position)
    
    def add_batch(self, entity_ids: List[str], embeddings: np.ndarray):
        """Add multiple embeddings to index."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        start_position = len(self.id_mapping)
        self.index.add(embeddings)
        
        for i, entity_id in enumerate(entity_ids):
            position = start_position + i
            self.id_mapping.append(entity_id)
            if entity_id not in self.entity_positions:
                self.entity_positions[entity_id] = []
            self.entity_positions[entity_id].append(position)
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for nearest neighbors."""
        if self.index.ntotal == 0:
            return []
        
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        
        query = query.reshape(1, -1)
        actual_k = min(k, self.index.ntotal)
        
        scores, indices = self.index.search(query, actual_k)
        
        results = []
        seen_entities = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_mapping):
                continue
            
            entity_id = self.id_mapping[idx]
            
            if entity_id not in seen_entities:
                results.append((entity_id, float(score)))
                seen_entities.add(entity_id)
        
        return results
    
    def remove_entity(self, entity_id: str):
        """Mark entity as removed."""
        if entity_id in self.entity_positions:
            del self.entity_positions[entity_id]
            for pos in self.entity_positions.get(entity_id, []):
                if pos < len(self.id_mapping):
                    self.id_mapping[pos] = ""
    
    def rebuild(self, entities: List[Tuple[str, np.ndarray]]):
        """Rebuild index from scratch."""
        self.index.reset()
        self.id_mapping.clear()
        self.entity_positions.clear()
        
        if not entities:
            return
        
        for entity_id, embedding in entities:
            self.add(entity_id, embedding)
    
    def save(self, path: Path):
        """Save index to disk."""
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))
        
        import json
        mapping_path = path.with_suffix('.mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump({
                'id_mapping': self.id_mapping,
                'entity_positions': self.entity_positions
            }, f)
    
    def load(self, path: Path) -> bool:
        """Load index from disk."""
        index_path = path.with_suffix('.faiss')
        mapping_path = path.with_suffix('.mapping.json')
        
        if not index_path.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(index_path))
            
            if mapping_path.exists():
                import json
                with open(mapping_path, 'r') as f:
                    data = json.load(f)
                    self.id_mapping = data['id_mapping']
                    self.entity_positions = data['entity_positions']
            
            return True
        except Exception as e:
            logger.error(f"Failed to load index from {path}: {e}")
            return False
    
    def __len__(self) -> int:
        return self.index.ntotal


class MultimodalSearchEngine:
    """
    Multimodal search engine with weighted fusion ranking.
    
    Configured for Danish's Smart Gallery models:
    - Face: 512-dim (InsightFace buffalo_l)
    - Image: 768-dim (CLIP ViT-L-14)
    - Text: 768-dim (CLIP ViT-L-14)
    """
    
    # Embedding dimensions matching your ml_pipeline_outputs
    FACE_DIM = 512   # InsightFace buffalo_l
    IMAGE_DIM = 768  # CLIP ViT-L-14
    TEXT_DIM = 768   # CLIP ViT-L-14 text encoder
    DINO_DIM = 1024  # DINOv2 ViT-L-14 (optional)
    
    def __init__(
        self,
        index_dir: Path,
        entity_store: EntityStore,
        embedding_engine: EmbeddingEngine,
        fusion_weights: Optional[Dict[str, float]] = None,
        use_dino: bool = False
    ):
        """
        Initialize search engine.
        
        Args:
            index_dir: Directory for storing FAISS indexes
            entity_store: Entity storage instance
            embedding_engine: Embedding generation instance
            fusion_weights: Default fusion weights for ranking
            use_dino: Whether to use DINOv2 index
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.entity_store = entity_store
        self.embedding_engine = embedding_engine
        self.use_dino = use_dino
        
        # Default fusion weights
        self.default_weights = fusion_weights or {
            "face": 0.5,
            "image": 0.3,
            "text": 0.2
        }
        
        # Initialize indexes with correct dimensions
        self.face_index = FAISSIndex(self.FACE_DIM)
        self.image_index = FAISSIndex(self.IMAGE_DIM)
        self.text_index = FAISSIndex(self.TEXT_DIM)
        
        # Optional DINOv2 index
        self.dino_index = FAISSIndex(self.DINO_DIM) if use_dino else None
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize search engine and load existing indexes."""
        if self._initialized:
            return
        
        # Try to load existing indexes
        face_loaded = self.face_index.load(self.index_dir / "face_index")
        image_loaded = self.image_index.load(self.index_dir / "image_index")
        text_loaded = self.text_index.load(self.index_dir / "text_index")
        
        if face_loaded:
            logger.info(f"Loaded face index with {len(self.face_index)} vectors (512-dim)")
        if image_loaded:
            logger.info(f"Loaded image index with {len(self.image_index)} vectors (768-dim)")
        if text_loaded:
            logger.info(f"Loaded text index with {len(self.text_index)} vectors (768-dim)")
        
        if self.use_dino and self.dino_index:
            dino_loaded = self.dino_index.load(self.index_dir / "dino_index")
            if dino_loaded:
                logger.info(f"Loaded DINOv2 index with {len(self.dino_index)} vectors (1024-dim)")
        
        # If no indexes loaded, rebuild from entity store
        if not (face_loaded or image_loaded or text_loaded):
            logger.info("No existing indexes found, building from entity store...")
            await self._rebuild_all_indexes()
        
        self._initialized = True
    
    async def _rebuild_all_indexes(self):
        """Rebuild all indexes from entity store."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._rebuild_indexes_sync)
    
    def _rebuild_indexes_sync(self):
        """Synchronous index rebuild."""
        face_data = []
        image_data = []
        text_data = []
        dino_data = []
        
        for entity in self.entity_store.list_entities():
            if entity.face_embeddings:
                for emb in entity.face_embeddings:
                    face_data.append((entity.entity_id, np.array(emb)))
            
            if entity.image_embeddings:
                for emb in entity.image_embeddings:
                    image_data.append((entity.entity_id, np.array(emb)))
            
            if entity.text_embedding:
                text_data.append((entity.entity_id, np.array(entity.text_embedding)))
            
            # Handle DINOv2 embeddings if stored
            if self.use_dino and hasattr(entity, 'dino_embeddings') and entity.dino_embeddings:
                for emb in entity.dino_embeddings:
                    dino_data.append((entity.entity_id, np.array(emb)))
        
        self.face_index.rebuild(face_data)
        self.image_index.rebuild(image_data)
        self.text_index.rebuild(text_data)
        
        if self.use_dino and self.dino_index:
            self.dino_index.rebuild(dino_data)
        
        logger.info(f"Rebuilt indexes: face={len(face_data)}, image={len(image_data)}, text={len(text_data)}")
    
    def save_indexes(self):
        """Save all indexes to disk."""
        self.face_index.save(self.index_dir / "face_index")
        self.image_index.save(self.index_dir / "image_index")
        self.text_index.save(self.index_dir / "text_index")
        
        if self.use_dino and self.dino_index:
            self.dino_index.save(self.index_dir / "dino_index")
        
        logger.info("All indexes saved")
    
    def get_index_sizes(self) -> Dict[str, int]:
        """Get number of vectors in each index."""
        sizes = {
            "face": len(self.face_index),
            "image": len(self.image_index),
            "text": len(self.text_index)
        }
        if self.use_dino and self.dino_index:
            sizes["dino"] = len(self.dino_index)
        return sizes
    
    def get_face_count(self) -> int:
        return len(self.face_index)
    
    def get_image_count(self) -> int:
        return len(self.image_index)
    
    # =========================================================================
    # Indexing Operations
    # =========================================================================
    
    async def index_entity(self, entity: Entity):
        """Index an entity's embeddings."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._index_entity_sync, entity)
    
    def _index_entity_sync(self, entity: Entity):
        """Synchronous entity indexing."""
        # Index face embeddings (512-dim)
        if entity.face_embeddings:
            for emb in entity.face_embeddings:
                self.face_index.add(entity.entity_id, np.array(emb))
        
        # Index image embeddings (768-dim CLIP)
        if entity.image_embeddings:
            for emb in entity.image_embeddings:
                self.image_index.add(entity.entity_id, np.array(emb))
        
        # Index text embedding (768-dim CLIP)
        if entity.text_embedding:
            self.text_index.add(entity.entity_id, np.array(entity.text_embedding))
    
    async def remove_entity(self, entity_id: str):
        """Remove entity from all indexes."""
        self.face_index.remove_entity(entity_id)
        self.image_index.remove_entity(entity_id)
        self.text_index.remove_entity(entity_id)
        if self.dino_index:
            self.dino_index.remove_entity(entity_id)
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    async def search_by_face(
        self,
        image_bytes: bytes,
        top_k: int = 10,
        fusion_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search entities by face image.
        
        Flow: Face → Entity → All Related Images
        """
        # Generate face embedding (512-dim)
        face_embedding = await self.embedding_engine.embed_face(image_bytes)
        
        if face_embedding is None:
            raise ValueError("No face detected in query image")
        
        # Also generate image embedding for fusion (768-dim)
        image_embedding = await self.embedding_engine.embed_image(image_bytes)
        
        # Search face index (primary)
        face_results = await self._search_index(
            self.face_index, face_embedding, top_k * 3
        )
        
        # Search image index (secondary)
        image_results = await self._search_index(
            self.image_index, image_embedding, top_k * 2
        )
        
        # Fuse results with face-weighted scoring
        weights = fusion_weights or {"face": 0.7, "image": 0.3, "text": 0.0}
        
        return self._fuse_and_rank(
            face_results=face_results,
            image_results=image_results,
            text_results=[],
            weights=weights,
            top_k=top_k,
            query_type="face"
        )
    
    async def search_by_image(
        self,
        image_bytes: bytes,
        top_k: int = 10,
        fusion_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search entities by general image.
        
        Flow: Image → Similar Scenes + Extracted Faces
        """
        # Generate image embedding (768-dim CLIP)
        image_embedding = await self.embedding_engine.embed_image(image_bytes)
        
        # Try to extract face embedding (512-dim)
        face_embedding = await self.embedding_engine.embed_face(image_bytes)
        
        # Search image index (primary)
        image_results = await self._search_index(
            self.image_index, image_embedding, top_k * 3
        )
        
        # Search face index if face detected
        face_results = []
        if face_embedding is not None:
            face_results = await self._search_index(
                self.face_index, face_embedding, top_k * 2
            )
        
        # Fuse results with image-weighted scoring
        weights = fusion_weights or {"face": 0.3, "image": 0.6, "text": 0.1}
        
        return self._fuse_and_rank(
            face_results=face_results,
            image_results=image_results,
            text_results=[],
            weights=weights,
            top_k=top_k,
            query_type="image"
        )
    
    async def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        fusion_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search entities by text description.
        
        Flow: Text → Images + Faces (semantic search)
        """
        # Generate text embedding (768-dim CLIP)
        text_embedding = await self.embedding_engine.embed_text(query)
        
        # Search text index
        text_results = await self._search_index(
            self.text_index, text_embedding, top_k * 2
        )
        
        # Also search image index (CLIP text-image compatibility)
        image_results = await self._search_index(
            self.image_index, text_embedding, top_k * 2
        )
        
        # Fuse results with text-weighted scoring
        weights = fusion_weights or {"face": 0.1, "image": 0.4, "text": 0.5}
        
        return self._fuse_and_rank(
            face_results=[],
            image_results=image_results,
            text_results=text_results,
            weights=weights,
            top_k=top_k,
            query_type="text"
        )
    
    async def _search_index(
        self,
        index: FAISSIndex,
        query: np.ndarray,
        k: int
    ) -> List[Tuple[str, float]]:
        """Search a single index."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, 
            index.search, 
            query, 
            k
        )
    
    def _fuse_and_rank(
        self,
        face_results: List[Tuple[str, float]],
        image_results: List[Tuple[str, float]],
        text_results: List[Tuple[str, float]],
        weights: Dict[str, float],
        top_k: int,
        query_type: str
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple modalities and rank.
        
        Uses weighted score fusion:
        final_score = w_face * face_sim + w_image * image_sim + w_text * text_sim
        """
        entity_scores: Dict[str, Dict[str, float]] = {}
        
        for entity_id, score in face_results:
            if entity_id not in entity_scores:
                entity_scores[entity_id] = {"face": 0.0, "image": 0.0, "text": 0.0}
            entity_scores[entity_id]["face"] = max(entity_scores[entity_id]["face"], score)
        
        for entity_id, score in image_results:
            if entity_id not in entity_scores:
                entity_scores[entity_id] = {"face": 0.0, "image": 0.0, "text": 0.0}
            entity_scores[entity_id]["image"] = max(entity_scores[entity_id]["image"], score)
        
        for entity_id, score in text_results:
            if entity_id not in entity_scores:
                entity_scores[entity_id] = {"face": 0.0, "image": 0.0, "text": 0.0}
            entity_scores[entity_id]["text"] = max(entity_scores[entity_id]["text"], score)
        
        # Calculate fused scores
        fused_results = []
        for entity_id, scores in entity_scores.items():
            fused_score = (
                weights.get("face", 0) * scores["face"] +
                weights.get("image", 0) * scores["image"] +
                weights.get("text", 0) * scores["text"]
            )
            
            modalities_matched = sum(1 for s in scores.values() if s > 0)
            confidence = modalities_matched / 3.0
            
            fused_results.append({
                "entity_id": entity_id,
                "fused_score": fused_score,
                "confidence": confidence,
                "match_details": scores
            })
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
        
        # Get top k and enrich with entity data
        final_results = []
        for result in fused_results[:top_k]:
            entity = self.entity_store.get_entity(result["entity_id"])
            if entity:
                final_results.append({
                    "entity_id": result["entity_id"],
                    "score": round(result["fused_score"], 4),
                    "confidence": round(result["confidence"], 2),
                    "images": entity.images,
                    "faces": entity.faces,
                    "metadata": entity.metadata,
                    "match_details": {
                        k: round(v, 4) for k, v in result["match_details"].items()
                    }
                })
        
        return final_results
    
    # =========================================================================
    # Load Pre-computed Embeddings from Google Drive
    # =========================================================================
    
    async def load_precomputed_embeddings(
        self,
        face_embeddings_path: str,
        image_embeddings_clip_path: str,
        image_metadata_path: str,
        faiss_clip_path: Optional[str] = None,
        faiss_dino_path: Optional[str] = None
    ):
        """
        Load pre-computed embeddings from your ml_pipeline_outputs.
        
        Args:
            face_embeddings_path: Path to face_embeddings.npy
            image_embeddings_clip_path: Path to image_embeddings_clip.npy
            image_metadata_path: Path to image_metadata.json
            faiss_clip_path: Optional path to faiss_clip.index
            faiss_dino_path: Optional path to faiss_dino.index
        """
        import json
        
        logger.info("Loading pre-computed embeddings from ml_pipeline_outputs...")
        
        # Load face embeddings
        if Path(face_embeddings_path).exists():
            face_embs = np.load(face_embeddings_path)
            logger.info(f"Loaded {len(face_embs)} face embeddings")
        
        # Load CLIP image embeddings
        if Path(image_embeddings_clip_path).exists():
            clip_embs = np.load(image_embeddings_clip_path)
            logger.info(f"Loaded {len(clip_embs)} CLIP image embeddings")
        
        # Load metadata
        if Path(image_metadata_path).exists():
            with open(image_metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(metadata)} images")
        
        # Optionally load pre-built FAISS indexes
        if faiss_clip_path and Path(faiss_clip_path).exists():
            self.image_index.index = faiss.read_index(faiss_clip_path)
            logger.info(f"Loaded pre-built CLIP FAISS index")
        
        logger.info("✅ Pre-computed embeddings loaded successfully")
