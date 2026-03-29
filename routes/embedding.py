from fastapi import APIRouter
from models import EmbeddingRequest, EmbeddingResponse
from dependencies import client

router = APIRouter()


@router.post("/embedding", response_model=EmbeddingResponse)
def embedding(body: EmbeddingRequest):
    vector = client.embeddings.create(
        input=body.text,
        model="text-embedding-3-large"
    ).data[0].embedding
    return EmbeddingResponse(embedding=vector)
