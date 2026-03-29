from fastapi import APIRouter
from models import Chunk, MessageRequest, MessageResponse
from dependencies import client, get_db_connection
from utils import cosine_similarity

router = APIRouter()

SIMILARITY_THRESHOLD = 0.5
CONTEXT_MAX_CHARS = 25000


@router.post("/ai", response_model=list[Chunk])
def ai(body: MessageRequest):

    # 1. enhance prompt

    # 2. convert to embedding
    embedding = client.embeddings.create(
        input=body.message,
        model="text-embedding-3-large"
    ).data[0].embedding

    # 3. retrieve top similar chunks
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, url, content, embedding FROM chunks WHERE embedding IS NOT NULL")
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row_id, url, content, embedding_json in rows:
        chunk_embedding = embedding_json
        sim = cosine_similarity(embedding, chunk_embedding)
        if sim >= SIMILARITY_THRESHOLD:
            results.append(Chunk(id=row_id, url=url, content=content, similarity=sim))

    results.sort(key=lambda x: x.similarity, reverse=True)

    # 3.5. build context
    context = ""
    for chunk in results:
        entry = f"{chunk.url}\n{chunk.content}\n\n"
        if len(context) + len(entry) > CONTEXT_MAX_CHARS:
            break
        context += entry
    context = context.rstrip("\n")

    # 4. generate answer

    return results
