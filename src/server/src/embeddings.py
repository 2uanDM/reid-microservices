import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from src.utils import Logger

load_dotenv()

logger = Logger(__name__)

GENDER_CONFIDENCE_THRESHOLD = float(os.getenv("GENDER_THRESHOLD", 0.9))


class PersonID:
    def __init__(
        self,
        fullbody_embedding: np.ndarray,
        fullbody_bbox: np.ndarray,
        body_conf: np.float32,
        gender: str = None,
        gender_confidence: float = 0.0,
    ):
        self.id = None
        self.fullbody_bbox = fullbody_bbox
        self.body_conf = body_conf
        self.ttl = 1000  # Frames

        # Gender metadata for filtering
        self.gender = gender  # "male" or "female"
        self.gender_confidence = gender_confidence

        # Only store the last (most recent) embedding - normalized
        self.last_embedding = self._normalize_embedding(fullbody_embedding)

        # Track when this person was last seen
        self.last_seen_frame = 0
        self.update_count = 1

    def set_id(self, id: int):
        self.id = id

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector for cosine similarity"""
        if embedding is None:
            return None
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def update_embedding(
        self, new_embedding: np.ndarray, body_score: float, frame_number: int = 0
    ):
        """
        Update the embedding only if the new detection has better confidence than the stored one.
        This ensures we keep the highest quality embedding (when person is most clearly visible).
        """
        if body_score > 0.75:  # Only consider confident detections
            # Only update if this detection is better than what we have stored
            if body_score > self.body_conf:
                self.last_embedding = self._normalize_embedding(new_embedding)
                self.body_conf = body_score  # Update to the new best confidence
                self.last_seen_frame = frame_number
                self.update_count += 1
                return True  # Indicate that embedding was updated
            else:
                self.last_seen_frame = frame_number
                self.update_count += 1
                return False  # Indicate that embedding was NOT updated
        return False  # Low confidence detection, nothing updated

    def get_embedding_for_comparison(self) -> np.ndarray:
        """Get the embedding to use for similarity comparison"""
        return self.last_embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert PersonID object to dictionary for Redis storage"""

        def safe_convert(obj):
            if obj is None:
                return None
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

        return {
            "id": safe_convert(self.id),
            "last_embedding": safe_convert(self.last_embedding),
            "fullbody_bbox": safe_convert(self.fullbody_bbox),
            "body_conf": float(self.body_conf) if self.body_conf is not None else None,
            "ttl": safe_convert(self.ttl),
            "last_seen_frame": safe_convert(self.last_seen_frame),
            "update_count": safe_convert(self.update_count),
            "gender": safe_convert(self.gender),
            "gender_confidence": safe_convert(self.gender_confidence),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonID":
        """Create PersonID object from dictionary"""

        def safe_convert_to_np(obj):
            if obj is None:
                return None
            elif isinstance(obj, list):
                return np.array(obj)
            return obj

        # Create instance with required fields
        instance = cls(
            fullbody_embedding=safe_convert_to_np(data["last_embedding"]),
            fullbody_bbox=safe_convert_to_np(data["fullbody_bbox"]),
            body_conf=np.float32(data["body_conf"]) if data["body_conf"] else None,
            gender=data.get("gender"),
            gender_confidence=data.get("gender_confidence", 0.0),
        )

        # Set all other fields
        instance.id = data["id"]
        instance.ttl = data["ttl"]
        instance.last_seen_frame = data.get("last_seen_frame", 0)
        instance.update_count = data.get("update_count", 1)

        return instance


class RedisPersonIDsStorage:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_prefix: str = "personid:",
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,
        )
        self.prefix = redis_prefix

    def _person_key(self, id: int) -> str:
        """Generate Redis key for person ID"""
        return f"{self.prefix}{id}"

    def _get_all_keys(self) -> List[str]:
        """Get all person keys in Redis"""
        return [key.decode() for key in self.redis_client.keys(f"{self.prefix}*")]

    def _get_all_ids(self) -> List[int]:
        """Get all person IDs in Redis"""
        keys = self._get_all_keys()
        return [int(key.replace(self.prefix, "")) for key in keys]

    def get_person_by_id(self, id: int) -> Optional[PersonID]:
        """Get person by ID from Redis"""
        key = self._person_key(id)
        data = self.redis_client.get(key)
        if data:
            person_dict = json.loads(data.decode())
            return PersonID.from_dict(person_dict)
        return None

    def add(self, person_id: PersonID):
        """Add person to Redis with TTL"""
        if person_id.id is None:
            raise ValueError("Person ID must be set before adding to storage")

        key = self._person_key(person_id.id)

        try:
            person_dict = person_id.to_dict()
            json_data = json.dumps(person_dict)
            self.redis_client.set(key, json_data)
            # Set expiration time based on TTL (converted to seconds)
            self.redis_client.expire(key, person_id.ttl)
        except Exception as e:
            logging.error(f"Error serializing person ID {person_id.id}: {str(e)}")
            logging.error(f"Person data: {person_id.__dict__}")
            raise

    def update(self, person_id: PersonID):
        """Update person in Redis"""
        self.add(person_id)

    def _calculate_cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings using PyTorch F.cosine_similarity.
        Returns a value between -1 and 1, where 1 is most similar (same as model serving).
        """
        # Convert to PyTorch tensors (same as model serving code)
        tensor1 = torch.tensor(embedding1, dtype=torch.float32)
        tensor2 = torch.tensor(embedding2, dtype=torch.float32)

        # Calculate cosine similarity using PyTorch F.cosine_similarity
        similarity = F.cosine_similarity(
            tensor1.unsqueeze(0), tensor2.unsqueeze(0)
        ).item()

        return similarity

    def search(
        self,
        current_person_id: PersonID,
        current_frame_id: list,
        threshold: float = 0.8,
    ) -> Tuple[Optional[PersonID], float]:
        """
        Search for the most similar person in Redis using cosine similarity.
        Now includes gender filtering - only searches among persons with the same gender.
        Uses PyTorch F.cosine_similarity (same as model serving).

        Args:
            current_person_id: The person to find matches for
            current_frame_id: List of IDs to exclude from search (current frame)
            threshold: Similarity threshold (-1 to 1, where 1 is identical)

        Returns:
            Tuple of (matched_person, similarity_score)
        """
        best_match = None
        best_similarity = -1.0  # Start with worst similarity

        current_embedding = current_person_id.get_embedding_for_comparison()
        if current_embedding is None:
            return None, -1.0

        current_gender = current_person_id.gender
        gender_filtered_count = 0
        total_persons_checked = 0

        # Check all person IDs in Redis
        for person_key in self._get_all_keys():
            person_id_value = int(person_key.replace(self.prefix, ""))

            # Skip persons that are in the current frame
            if person_id_value in current_frame_id:
                continue

            # Get person data from Redis
            data = self.redis_client.get(person_key)
            if not data:
                continue

            person_dict = json.loads(data.decode())
            person_id = PersonID.from_dict(person_dict)

            total_persons_checked += 1

            # Gender filtering: only compare with persons of the same gender if both have high confidence (>0.95)
            if (
                current_gender is not None
                and person_id.gender is not None
                and current_person_id.gender_confidence > GENDER_CONFIDENCE_THRESHOLD
                and person_id.gender_confidence > GENDER_CONFIDENCE_THRESHOLD
            ):
                if current_gender != person_id.gender:
                    logger.info(
                        f"Person {person_id.id}: skipped due to gender mismatch ({person_id.gender} vs {current_gender}) - both high confidence"
                    )
                    continue

            gender_filtered_count += 1

            stored_embedding = person_id.get_embedding_for_comparison()
            if stored_embedding is not None:
                # Calculate similarity using PyTorch F.cosine_similarity
                similarity = self._calculate_cosine_similarity(
                    current_embedding, stored_embedding
                )

                logger.info(
                    f"Person {person_id.id}: similarity = {similarity:.4f}, gender = {person_id.gender}"
                )

                if similarity > best_similarity and similarity >= threshold:
                    best_match = person_id
                    best_similarity = similarity

        logger.info(
            f"Gender filtering: {gender_filtered_count}/{total_persons_checked} persons processed (filtering applied when both confidence > {GENDER_CONFIDENCE_THRESHOLD})"
        )

        if best_match:
            confidence_info = ""
            if (
                current_person_id.gender_confidence > GENDER_CONFIDENCE_THRESHOLD
                and best_match.gender_confidence > GENDER_CONFIDENCE_THRESHOLD
            ):
                confidence_info = f", gender filtering applied (conf: {current_person_id.gender_confidence:.3f} vs {best_match.gender_confidence:.3f})"
            logger.info(
                f"Best match: Person {best_match.id} with similarity {best_similarity:.4f}, gender = {best_match.gender}{confidence_info}"
            )
        else:
            confidence_info = ""
            if current_person_id.gender_confidence > GENDER_CONFIDENCE_THRESHOLD:
                confidence_info = (
                    f" (query confidence: {current_person_id.gender_confidence:.3f})"
                )
            logger.info(
                f"No match found above threshold {threshold:.4f} with gender '{current_gender}'{confidence_info}"
            )

        return best_match, best_similarity

    def clear(self):
        """Clear all person IDs from Redis"""
        keys = self._get_all_keys()
        if keys:
            self.redis_client.delete(*keys)
