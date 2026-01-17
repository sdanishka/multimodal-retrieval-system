"""
Entity Store Module
===================
Entity-centric data storage for multimodal retrieval.

Each entity represents a unique subject (person, object, scene) with:
- Multiple face embeddings (if applicable)
- Multiple image embeddings
- Single text embedding (description)
- Metadata (tags, name, etc.)

Provides persistence via JSON serialization.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """
    Entity data structure.
    
    Represents a single entity in the retrieval system.
    One entity may have multiple embeddings per modality.
    """
    entity_id: str
    
    # Embedding storage (lists to support multiple per entity)
    face_embeddings: List[List[float]] = field(default_factory=list)
    image_embeddings: List[List[float]] = field(default_factory=list)
    text_embedding: Optional[List[float]] = None
    
    # Associated media paths
    images: List[str] = field(default_factory=list)
    faces: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary (embeddings excluded for API responses)."""
        return {
            "entity_id": self.entity_id,
            "images": self.images,
            "faces": self.faces,
            "metadata": self.metadata,
            "has_face_embedding": len(self.face_embeddings) > 0,
            "has_image_embedding": len(self.image_embeddings) > 0,
            "has_text_embedding": self.text_embedding is not None,
            "face_count": len(self.face_embeddings),
            "image_count": len(self.image_embeddings),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_full_dict(self) -> Dict[str, Any]:
        """Convert entity to full dictionary including embeddings."""
        return asdict(self)
    
    def add_face_embedding(self, embedding: np.ndarray):
        """Add a face embedding."""
        self.face_embeddings.append(embedding.tolist())
        self._update_timestamp()
    
    def add_image_embedding(self, embedding: np.ndarray):
        """Add an image embedding."""
        self.image_embeddings.append(embedding.tolist())
        self._update_timestamp()
    
    def set_text_embedding(self, embedding: np.ndarray):
        """Set the text embedding."""
        self.text_embedding = embedding.tolist()
        self._update_timestamp()
    
    def add_image_path(self, path: str):
        """Add an image path."""
        if path not in self.images:
            self.images.append(path)
            self._update_timestamp()
    
    def add_face_path(self, path: str):
        """Add a face crop path."""
        if path not in self.faces:
            self.faces.append(path)
            self._update_timestamp()
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Update metadata."""
        self.metadata.update(metadata)
        self._update_timestamp()
    
    def _update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow().isoformat()


class EntityStore:
    """
    Persistent entity storage.
    
    Manages entity creation, retrieval, and persistence.
    Uses JSON file for storage (can be replaced with database).
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize entity store.
        
        Args:
            storage_path: Path to JSON storage file
        """
        self.storage_path = Path(storage_path)
        self.entities: Dict[str, Entity] = {}
        self._dirty = False
    
    def __len__(self) -> int:
        return len(self.entities)
    
    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self.entities
    
    def load(self):
        """Load entities from storage."""
        if not self.storage_path.exists():
            logger.info(f"No existing entity store at {self.storage_path}")
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for entity_id, entity_data in data.items():
                self.entities[entity_id] = Entity(**entity_data)
            
            logger.info(f"Loaded {len(self.entities)} entities from storage")
        except Exception as e:
            logger.error(f"Failed to load entity store: {e}")
            raise
    
    def save(self):
        """Save entities to storage."""
        if not self._dirty:
            return
        
        try:
            # Create parent directory if needed
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                entity_id: entity.to_full_dict()
                for entity_id, entity in self.entities.items()
            }
            
            # Atomic write
            temp_path = self.storage_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_path.rename(self.storage_path)
            self._dirty = False
            
            logger.info(f"Saved {len(self.entities)} entities to storage")
        except Exception as e:
            logger.error(f"Failed to save entity store: {e}")
            raise
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def list_entities(self, limit: int = 100, offset: int = 0) -> List[Entity]:
        """List entities with pagination."""
        entity_list = list(self.entities.values())
        return entity_list[offset:offset + limit]
    
    def create_entity(
        self,
        entity_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Entity:
        """
        Create a new entity.
        
        Args:
            entity_id: Optional custom ID (auto-generated if not provided)
            metadata: Optional initial metadata
            
        Returns:
            New entity instance
        """
        if entity_id is None:
            entity_id = self._generate_entity_id()
        
        if entity_id in self.entities:
            raise ValueError(f"Entity {entity_id} already exists")
        
        entity = Entity(
            entity_id=entity_id,
            metadata=metadata or {}
        )
        
        self.entities[entity_id] = entity
        self._dirty = True
        
        return entity
    
    def create_or_update_entity(
        self,
        entity_id: str,
        face_embedding: Optional[np.ndarray] = None,
        image_embedding: Optional[np.ndarray] = None,
        text_embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Entity:
        """
        Create or update an entity.
        
        Args:
            entity_id: Entity identifier
            face_embedding: Optional face embedding to add
            image_embedding: Optional image embedding to add
            text_embedding: Optional text embedding to set
            metadata: Optional metadata to update
            
        Returns:
            Entity instance
        """
        if entity_id in self.entities:
            entity = self.entities[entity_id]
        else:
            entity = Entity(entity_id=entity_id)
            self.entities[entity_id] = entity
        
        if face_embedding is not None:
            entity.add_face_embedding(face_embedding)
        
        if image_embedding is not None:
            entity.add_image_embedding(image_embedding)
        
        if text_embedding is not None:
            entity.set_text_embedding(text_embedding)
        
        if metadata:
            entity.update_metadata(metadata)
        
        self._dirty = True
        return entity
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            True if deleted, False if not found
        """
        if entity_id in self.entities:
            del self.entities[entity_id]
            self._dirty = True
            return True
        return False
    
    def search_by_metadata(
        self,
        key: str,
        value: Any,
        limit: int = 100
    ) -> List[Entity]:
        """
        Search entities by metadata field.
        
        Args:
            key: Metadata key to search
            value: Value to match
            limit: Maximum results
            
        Returns:
            Matching entities
        """
        results = []
        for entity in self.entities.values():
            if key in entity.metadata and entity.metadata[key] == value:
                results.append(entity)
                if len(results) >= limit:
                    break
        return results
    
    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[Entity]:
        """
        Search entities by tags.
        
        Args:
            tags: Tags to search for
            match_all: If True, entity must have all tags
            
        Returns:
            Matching entities
        """
        results = []
        tag_set = set(tags)
        
        for entity in self.entities.values():
            entity_tags = set(entity.metadata.get("tags", []))
            
            if match_all:
                if tag_set.issubset(entity_tags):
                    results.append(entity)
            else:
                if tag_set.intersection(entity_tags):
                    results.append(entity)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_faces = sum(len(e.face_embeddings) for e in self.entities.values())
        total_images = sum(len(e.image_embeddings) for e in self.entities.values())
        total_text = sum(1 for e in self.entities.values() if e.text_embedding)
        
        return {
            "total_entities": len(self.entities),
            "total_face_embeddings": total_faces,
            "total_image_embeddings": total_images,
            "total_text_embeddings": total_text,
            "storage_path": str(self.storage_path)
        }
    
    def _generate_entity_id(self) -> str:
        """Generate a unique entity ID."""
        timestamp = datetime.utcnow().isoformat()
        content = f"{timestamp}_{len(self.entities)}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"entity_{hash_value}"
    
    def export_entities(self, entity_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export entities for backup or transfer.
        
        Args:
            entity_ids: Optional list of specific entities to export
            
        Returns:
            Export data dictionary
        """
        if entity_ids:
            entities = {
                eid: self.entities[eid].to_full_dict()
                for eid in entity_ids
                if eid in self.entities
            }
        else:
            entities = {
                eid: entity.to_full_dict()
                for eid, entity in self.entities.items()
            }
        
        return {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "entity_count": len(entities),
            "entities": entities
        }
    
    def import_entities(self, data: Dict[str, Any], overwrite: bool = False):
        """
        Import entities from export data.
        
        Args:
            data: Export data dictionary
            overwrite: If True, overwrite existing entities
        """
        entities_data = data.get("entities", {})
        
        for entity_id, entity_data in entities_data.items():
            if entity_id in self.entities and not overwrite:
                continue
            
            self.entities[entity_id] = Entity(**entity_data)
        
        self._dirty = True
        logger.info(f"Imported {len(entities_data)} entities")
