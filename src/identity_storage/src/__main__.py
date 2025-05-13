import os
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .identity_storage import IdentityStorage
from .schema import Person

load_dotenv()


# Request Models
class EmbeddingRequest(BaseModel):
    features: List[float] = Field(..., description="Embedding feature vector")
    detect_conf: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )


class SearchRequest(BaseModel):
    embedding: List[float] = Field(..., description="Query embedding vector")
    limit: int = Field(10, ge=1, description="Maximum number of results to return")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")


# Response Models
class SearchResult(BaseModel):
    id: int
    score: float


# FastAPI Application
app = FastAPI(
    title="Identity Storage API",
    description="API for managing and searching person embeddings",
    version="1.0.0",
)


# Dependency to get the identity storage instance
async def get_identity_storage():
    # Get configuration from environment variables
    collection_name = os.getenv("QDRANT_COLLECTION", "persons")
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    vector_size = int(os.getenv("VECTOR_SIZE", "512"))
    top_k = int(os.getenv("TOP_K", "5"))
    conf_threshold = float(os.getenv("CONF_THRESHOLD", "0.7"))

    storage = IdentityStorage(
        collection_name=collection_name,
        host=host,
        port=port,
        vector_size=vector_size,
        top_k=top_k,
        conf_threshold=conf_threshold,
    )

    # Initialize collection
    await storage.init_collection()
    return storage


@app.post("/persons/{person_id}", response_model=bool, status_code=201)
async def insert_person(
    request: EmbeddingRequest,
    person_id: int = Path(..., description="ID of the person"),
    storage: IdentityStorage = Depends(get_identity_storage),
):
    """Insert or update a person in the storage"""
    try:
        result = await storage.insert(
            person_id=person_id,
            features=request.features,
            detect_conf=request.detect_conf,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persons/{person_id}", response_model=Optional[Person])
async def get_person(
    person_id: int = Path(..., description="ID of the person to retrieve"),
    storage: IdentityStorage = Depends(get_identity_storage),
):
    """Get a person by ID"""
    try:
        person = await storage.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        return person
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/persons/{person_id}", response_model=bool)
async def delete_person(
    person_id: int = Path(..., description="ID of the person to delete"),
    storage: IdentityStorage = Depends(get_identity_storage),
):
    """Delete a person by ID"""
    try:
        result = await storage.delete(person_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[SearchResult])
async def search_persons(
    request: SearchRequest,
    storage: IdentityStorage = Depends(get_identity_storage),
):
    """Search for similar persons using cosine similarity"""
    try:
        results = await storage.search(
            embedding=request.embedding,
            limit=request.limit,
            threshold=request.threshold,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/empty", response_model=bool)
async def empty(storage: IdentityStorage = Depends(get_identity_storage)):
    """Empty the collection"""
    try:
        await storage.empty()
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


# Custom exception handler for HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.detail},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("APP_PORT", "8000")))
