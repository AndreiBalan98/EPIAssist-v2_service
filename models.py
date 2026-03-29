from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]


class SimilarityRequest(BaseModel):
    strings: list[str]


class SimilarityPair(BaseModel):
    string1: str
    string2: str
    similarity: float


class SimilarityResponse(BaseModel):
    results: list[SimilarityPair]


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    message: str
