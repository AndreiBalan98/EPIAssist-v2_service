from fastapi import APIRouter
from models import MessageRequest, MessageResponse
from dependencies import client

router = APIRouter()


@router.post("/ai", response_model=MessageResponse)
def ai(body: MessageRequest):

    # 1. enhance prompt

    # 2. convert to embedding
    embedding = client.embeddings.create(
        input=body.message,
        model="text-embedding-3-large"
    ).data[0].embedding

    # 3. retrieve top similar chunks

    # 4. generate answer

    return MessageResponse(message=str(embedding))
