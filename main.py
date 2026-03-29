import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from itertools import combinations
import numpy as np

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
client = OpenAI(api_key=api_key)

class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    message: str


class SimilarityRequest(BaseModel):
    strings: list[str]


class SimilarityPair(BaseModel):
    string1: str
    string2: str
    similarity: float


class SimilarityResponse(BaseModel):
    results: list[SimilarityPair]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(body: SimilarityRequest):
    embeddings_data = client.embeddings.create(
        input=body.strings,
        model="text-embedding-3-large" #text-embedding-3-small
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


@app.post("/ai", response_model=MessageResponse)
def ai(body: MessageRequest):
    
    #1. enhance prompt

    #2. convert to embedding
    embedding = client.embeddings.create(
        input=body.message,
        model="text-embedding-3-large" #text-embedding-3-small
    ).data[0].embedding

    #3. retrieve top similar chunks
    

    #4. generate answer


    return MessageResponse(message=str(embedding))
