from pydantic import BaseModel


class Embedding(BaseModel):
    value: list[float]
    detect_conf: float


class Person(BaseModel):
    id: int
    first_embedding: Embedding | None
    last_embedding: Embedding | None
    top_k_embeddings: list[Embedding]


__all__ = ["Person", "Embedding"]
