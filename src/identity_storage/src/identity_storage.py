from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    CollectionStatus,
    Distance,
    PointStruct,
    VectorParams,
)
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .schema import Embedding, Person

load_dotenv()


class IdentityStorage:
    def __init__(
        self,
        collection_name: str = "persons",
        host: str = "localhost",
        port: int = 6333,
        vector_size: int = 512,
        top_k: int = 5,
        conf_threshold: float = 0.7,
    ):
        self.client = AsyncQdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Top k embeddings
        self.top_k = top_k
        self.conf_threshold = conf_threshold

    async def init_collection(self):
        """Initialize collection if it doesn't exist"""
        collections_response = await self.client.get_collections()
        collections = collections_response.collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if not collection_exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def insert(
        self,
        person_id: int,
        features: list[float],
        detect_conf: float,
    ) -> bool:
        """
        Insert or update a person in the storage

        Args:
            person_id: ID of the person to insert
            features: List of features to insert
            detect_conf: Detection confidence score

        Returns:
            bool: True if successful
        """

        if detect_conf < 0.7:
            print(
                f"Detection confidence {detect_conf} is too low for person {person_id}. Skipping..."
            )
            return False

        # Normalize featuress
        features = np.array(features)
        features = features / np.linalg.norm(features)

        embedding = Embedding(
            value=features,
            detect_conf=detect_conf,
        )

        # Check if person already exists
        existing_person = await self.get(person_id)

        if existing_person is None:
            existing_person = Person(
                id=person_id,
                first_embedding=embedding,
                last_embedding=embedding,
                top_k_embeddings=[embedding],
            )
        else:
            existing_person.last_embedding = embedding

            if len(existing_person.top_k_embeddings) < self.top_k:
                existing_person.top_k_embeddings.append(embedding)
            else:
                # Find the embedding with the lowest confidence
                min_conf_idx = min(
                    range(len(existing_person.top_k_embeddings)),
                    key=lambda i: existing_person.top_k_embeddings[i].detect_conf,
                )
                if (
                    embedding.detect_conf
                    > existing_person.top_k_embeddings[min_conf_idx].detect_conf
                ):
                    existing_person.top_k_embeddings[min_conf_idx] = embedding

        # Calculating the avg embedding first + last + top_k
        avg_embedding = np.mean(
            [
                emb.value
                for emb in existing_person.top_k_embeddings
                + [existing_person.first_embedding, existing_person.last_embedding]
            ],
            axis=0,
        )

        # Create point
        point = PointStruct(
            id=person_id,
            vector=avg_embedding,
            payload={
                "first_embedding": existing_person.first_embedding.model_dump()
                if existing_person.first_embedding
                else None,
                "last_embedding": existing_person.last_embedding.model_dump()
                if existing_person.last_embedding
                else None,
                "top_k_embeddings": [
                    emb.model_dump() for emb in existing_person.top_k_embeddings
                ],
            },
        )

        # Upsert the point
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

        return True

    async def get(self, person_id: int) -> Optional[Person]:
        """
        Get a person by ID

        Args:
            person_id: ID of the person to retrieve

        Returns:
            Person object if found, None otherwise
        """
        # Search for person with the given ID
        response = await self.client.retrieve(
            collection_name=self.collection_name, ids=[person_id]
        )

        if not response:
            return None

        point = response[0]

        # Reconstruct person
        person = Person(
            id=point.id,
            first_embedding=Embedding(**point.payload.get("first_embedding"))
            if point.payload.get("first_embedding")
            else None,
            last_embedding=Embedding(**point.payload.get("last_embedding"))
            if point.payload.get("last_embedding")
            else None,
            top_k_embeddings=[
                Embedding(**emb) for emb in point.payload.get("top_k_embeddings")
            ],
        )

        return person

    async def delete(self, person_id: int) -> bool:
        """
        Delete a person by ID

        Args:
            person_id: ID of the person to delete

        Returns:
            bool: True if successful
        """
        result = await self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=person_id))]
            ),
        )

        return result.status == CollectionStatus.GREEN

    async def search(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> List[Dict]:
        """
        Search for similar persons using cosine similarity

        Args:
            embedding: The query embedding
            limit: Maximum number of results
            threshold: Similarity threshold (0-1)

        Returns:
            List of dictionaries containing person ID and similarity score
        """
        # Normalize embedding
        embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding)

        # Search
        search_result = await self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=limit,
            score_threshold=threshold,
        )

        # Format results
        results = []
        for hit in search_result:
            results.append({"id": hit.id, "score": hit.score})

        return results

    async def empty(self):
        """
        Empty the collection
        """
        await self.client.delete_collection(collection_name=self.collection_name)
        await self.init_collection()
