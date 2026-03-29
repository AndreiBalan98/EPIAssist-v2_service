from fastapi import APIRouter
from itertools import combinations
from models import SimilarityRequest, SimilarityResponse, SimilarityPair
from dependencies import client
from utils import cosine_similarity

router = APIRouter()


@router.post("/similarity", response_model=SimilarityResponse)
def similarity(body: SimilarityRequest):
    embeddings_data = client.embeddings.create(
        input=body.strings,
        model="text-embedding-3-large"
    ).data
    embeddings = [e.embedding for e in embeddings_data]

    pairs = [
        SimilarityPair(
            string1=body.strings[i],
            string2=body.strings[j],
            similarity=cosine_similarity(embeddings[i], embeddings[j])
        )
        for i, j in combinations(range(len(body.strings)), 2)
    ]
    pairs.sort(key=lambda x: x.similarity, reverse=True)

    return SimilarityResponse(results=pairs)
