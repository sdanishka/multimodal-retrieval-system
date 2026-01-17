"""
Embedding Engine Module - Using Danish's Smart Gallery Models
=============================================================
Configured to use the exact models from the ml_pipeline_outputs project:

- Face Embeddings: InsightFace buffalo_l (ArcFace) - 512-dim
- Image Embeddings: CLIP ViT-L-14 - 768-dim  
- Image Embeddings: DINOv2 ViT-L-14 - 1024-dim (optional)
- Object Detection: Your fine-tuned YOLOv8x model

All embeddings are normalized for cosine similarity search.
"""

import io
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=4)


class EmbeddingEngine:
    """
    Unified embedding engine using Danish's Smart Gallery models.
    
    Models:
    - InsightFace buffalo_l (ArcFace) for face recognition
    - CLIP ViT-L-14 for image/text embeddings
    - DINOv2 ViT-L-14 for visual similarity (optional)
    - Custom YOLOv8x for object detection
    """
    
    # Model configurations matching your ml_pipeline_outputs
    FACE_CONFIG = {
        'model_name': 'buffalo_l',
        'det_thresh': 0.4,
        'min_face_size': 30,
        'det_size': (640, 640),
        'embedding_dim': 512,
    }
    
    CLIP_CONFIG = {
        'model_name': 'ViT-L-14',
        'pretrained': 'openai',
        'embedding_dim': 768,
        'image_size': 224,
    }
    
    DINO_CONFIG = {
        'model_name': 'dinov2_vitl14',
        'embedding_dim': 1024,
        'image_size': 518,
    }
    
    def __init__(
        self,
        use_dino: bool = False,
        yolo_model_path: Optional[str] = None,
        google_drive_base: Optional[str] = None
    ):
        """
        Initialize embedding engine.
        
        Args:
            use_dino: Whether to also use DINOv2 for image embeddings
            yolo_model_path: Path to your fine-tuned YOLO model
            google_drive_base: Base path to ml_pipeline_outputs in Google Drive
        """
        self.use_dino = use_dino
        self.yolo_model_path = yolo_model_path
        self.google_drive_base = google_drive_base
        
        # Model instances
        self.face_analyzer = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.dino_model = None
        self.dino_transform = None
        self.yolo_model = None
        
        self.device = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize all embedding models."""
        if self._initialized:
            return
            
        logger.info("Initializing embedding models (Smart Gallery config)...")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._load_models)
        
        self._initialized = True
        logger.info("All embedding models initialized")
    
    def _load_models(self):
        """Load all ML models (runs in thread pool)."""
        import torch
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load InsightFace for face embeddings
        self._load_insightface()
        
        # Load CLIP ViT-L-14 for image/text embeddings
        self._load_clip()
        
        # Optionally load DINOv2
        if self.use_dino:
            self._load_dino()
        
        # Load your fine-tuned YOLO model
        if self.yolo_model_path:
            self._load_yolo()
    
    def _load_insightface(self):
        """Load InsightFace buffalo_l model for face embeddings."""
        try:
            from insightface.app import FaceAnalysis
            
            logger.info(f"Loading InsightFace {self.FACE_CONFIG['model_name']}...")
            
            self.face_analyzer = FaceAnalysis(
                name=self.FACE_CONFIG['model_name'],
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(
                ctx_id=0 if self.device == "cuda" else -1,
                det_size=self.FACE_CONFIG['det_size']
            )
            
            logger.info(f"✅ InsightFace loaded (buffalo_l/ArcFace, {self.FACE_CONFIG['embedding_dim']}-dim)")
            
        except Exception as e:
            logger.error(f"InsightFace loading failed: {e}")
            raise RuntimeError("InsightFace is required for face embeddings")
    
    def _load_clip(self):
        """Load CLIP ViT-L-14 model for image and text embeddings."""
        try:
            import open_clip
            
            logger.info(f"Loading CLIP {self.CLIP_CONFIG['model_name']}...")
            
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.CLIP_CONFIG['model_name'],
                pretrained=self.CLIP_CONFIG['pretrained'],
                device=self.device
            )
            self.clip_model.eval()
            
            self.clip_tokenizer = open_clip.get_tokenizer(self.CLIP_CONFIG['model_name'])
            
            logger.info(f"✅ CLIP loaded (ViT-L-14, {self.CLIP_CONFIG['embedding_dim']}-dim)")
            
        except ImportError:
            # Fallback to original CLIP if open_clip not available
            logger.warning("open_clip not found, trying original CLIP...")
            self._load_clip_fallback()
        except Exception as e:
            logger.error(f"CLIP loading failed: {e}")
            raise RuntimeError("CLIP model is required for image/text embeddings")
    
    def _load_clip_fallback(self):
        """Fallback to OpenAI CLIP if open_clip not available."""
        import clip
        
        # Note: Original CLIP only has ViT-B/32, ViT-B/16, ViT-L/14
        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-L/14",  # Closest to your config
            device=self.device
        )
        self.clip_model.eval()
        self.clip_tokenizer = clip.tokenize
        
        # Update config for fallback
        self.CLIP_CONFIG['embedding_dim'] = 768
        
        logger.info(f"✅ CLIP loaded via fallback (ViT-L/14, 768-dim)")
    
    def _load_dino(self):
        """Load DINOv2 ViT-L-14 model for visual embeddings."""
        try:
            import torch
            
            logger.info(f"Loading DINOv2 {self.DINO_CONFIG['model_name']}...")
            
            self.dino_model = torch.hub.load(
                'facebookresearch/dinov2',
                self.DINO_CONFIG['model_name']
            )
            self.dino_model = self.dino_model.to(self.device)
            self.dino_model.eval()
            
            # DINOv2 transform
            from torchvision import transforms
            self.dino_transform = transforms.Compose([
                transforms.Resize(self.DINO_CONFIG['image_size'], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.DINO_CONFIG['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            logger.info(f"✅ DINOv2 loaded ({self.DINO_CONFIG['model_name']}, {self.DINO_CONFIG['embedding_dim']}-dim)")
            
        except Exception as e:
            logger.warning(f"DINOv2 loading failed: {e}")
            self.use_dino = False
    
    def _load_yolo(self):
        """Load your fine-tuned YOLOv8x model."""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO from {self.yolo_model_path}...")
            
            self.yolo_model = YOLO(self.yolo_model_path)
            
            logger.info(f"✅ YOLO loaded (your fine-tuned model)")
            
        except Exception as e:
            logger.warning(f"YOLO loading failed: {e}")
            self.yolo_model = None
    
    def get_loaded_models(self) -> List[str]:
        """Return list of successfully loaded models."""
        models = []
        if self.face_analyzer is not None:
            models.append(f"insightface_{self.FACE_CONFIG['model_name']}")
        if self.clip_model is not None:
            models.append(f"clip_{self.CLIP_CONFIG['model_name']}")
        if self.dino_model is not None:
            models.append(f"dino_{self.DINO_CONFIG['model_name']}")
        if self.yolo_model is not None:
            models.append("yolov8x_finetuned")
        return models
    
    def get_embedding_dims(self) -> Dict[str, int]:
        """Return embedding dimensions for each modality."""
        return {
            'face': self.FACE_CONFIG['embedding_dim'],
            'image_clip': self.CLIP_CONFIG['embedding_dim'],
            'image_dino': self.DINO_CONFIG['embedding_dim'] if self.use_dino else 0,
            'text': self.CLIP_CONFIG['embedding_dim'],
        }
    
    # =========================================================================
    # Face Embeddings (InsightFace buffalo_l)
    # =========================================================================
    
    async def embed_face(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Generate face embedding from image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            512-dimensional normalized face embedding, or None if no face detected
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._embed_face_sync, image_bytes)
    
    def _embed_face_sync(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Synchronous face embedding generation."""
        try:
            import cv2
            
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to BGR for InsightFace
            img_array = np.array(image)[:, :, ::-1]
            
            # Detect faces
            faces = self.face_analyzer.get(img_array)
            
            if not faces:
                logger.debug("No faces detected in image")
                return None
            
            # Get best face by detection score
            best_face = max(faces, key=lambda f: f.det_score)
            
            # Check detection threshold
            if best_face.det_score < self.FACE_CONFIG['det_thresh']:
                logger.debug(f"Face score {best_face.det_score} below threshold {self.FACE_CONFIG['det_thresh']}")
                return None
            
            embedding = best_face.embedding
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Face embedding error: {e}")
            raise
    
    async def detect_and_embed_faces(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Detect all faces in image and return embeddings with metadata.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            List of dicts with 'embedding', 'bbox', 'det_score', 'age', 'gender'
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._detect_faces_sync, image_bytes)
    
    def _detect_faces_sync(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Synchronous face detection and embedding."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)[:, :, ::-1]  # RGB to BGR
            
            faces = self.face_analyzer.get(img_array)
            
            results = []
            for face in faces:
                if face.det_score < self.FACE_CONFIG['det_thresh']:
                    continue
                
                if face.embedding is None:
                    continue
                
                embedding = face.embedding / np.linalg.norm(face.embedding)
                
                results.append({
                    'embedding': embedding.astype(np.float32),
                    'bbox': face.bbox.astype(int).tolist(),
                    'det_score': float(face.det_score),
                    'age': int(face.age) if hasattr(face, 'age') and face.age else None,
                    'gender': 'M' if hasattr(face, 'gender') and face.gender == 1 else 'F' if hasattr(face, 'gender') else None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    # =========================================================================
    # Image Embeddings (CLIP ViT-L-14)
    # =========================================================================
    
    async def embed_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Generate CLIP image embedding.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            768-dimensional normalized image embedding
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._embed_image_sync, image_bytes)
    
    def _embed_image_sync(self, image_bytes: bytes) -> np.ndarray:
        """Synchronous CLIP image embedding generation."""
        import torch
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
            
            embedding = image_features.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"CLIP image embedding error: {e}")
            raise
    
    async def embed_image_dino(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Generate DINOv2 image embedding.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            1024-dimensional normalized image embedding, or None if DINOv2 not loaded
        """
        if not self.use_dino or self.dino_model is None:
            return None
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._embed_image_dino_sync, image_bytes)
    
    def _embed_image_dino_sync(self, image_bytes: bytes) -> np.ndarray:
        """Synchronous DINOv2 image embedding generation."""
        import torch
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_input = self.dino_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.dino_model(image_input)
            
            embedding = features.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"DINOv2 image embedding error: {e}")
            raise
    
    # =========================================================================
    # Text Embeddings (CLIP ViT-L-14)
    # =========================================================================
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate CLIP text embedding.
        
        Args:
            text: Text description
            
        Returns:
            768-dimensional normalized text embedding (same space as CLIP images)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._embed_text_sync, text)
    
    def _embed_text_sync(self, text: str) -> np.ndarray:
        """Synchronous text embedding generation."""
        import torch
        
        try:
            text_tokens = self.clip_tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
            
            embedding = text_features.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Text embedding error: {e}")
            raise
    
    # =========================================================================
    # Object Detection (Your fine-tuned YOLOv8x)
    # =========================================================================
    
    async def detect_objects(
        self,
        image_bytes: bytes,
        confidence: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Detect objects using your fine-tuned YOLO model.
        
        Args:
            image_bytes: Raw image bytes
            confidence: Detection confidence threshold
            
        Returns:
            List of detections with 'class', 'confidence', 'bbox'
        """
        if self.yolo_model is None:
            return []
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, 
            self._detect_objects_sync, 
            image_bytes, 
            confidence
        )
    
    def _detect_objects_sync(self, image_bytes: bytes, confidence: float) -> List[Dict[str, Any]]:
        """Synchronous object detection."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            results = self.yolo_model(image, conf=confidence, verbose=False)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'class': result.names[int(box.cls)],
                        'class_id': int(box.cls),
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist()
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    async def embed_images_batch(
        self, 
        image_bytes_list: List[bytes], 
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """Batch process multiple images with CLIP."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, 
            self._embed_images_batch_sync, 
            image_bytes_list, 
            batch_size
        )
    
    def _embed_images_batch_sync(
        self, 
        image_bytes_list: List[bytes], 
        batch_size: int
    ) -> List[np.ndarray]:
        """Synchronous batch image embedding."""
        import torch
        
        all_embeddings = []
        
        for i in range(0, len(image_bytes_list), batch_size):
            batch = image_bytes_list[i:i + batch_size]
            
            images = []
            for img_bytes in batch:
                image = Image.open(io.BytesIO(img_bytes))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                images.append(self.clip_preprocess(image))
            
            image_input = torch.stack(images).to(self.device)
            
            with torch.no_grad():
                features = self.clip_model.encode_image(image_input)
            
            for feat in features:
                emb = feat.cpu().numpy().flatten()
                emb = emb / np.linalg.norm(emb)
                all_embeddings.append(emb.astype(np.float32))
        
        return all_embeddings
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))
    
    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute Euclidean distance between two embeddings."""
        return float(np.linalg.norm(emb1 - emb2))
