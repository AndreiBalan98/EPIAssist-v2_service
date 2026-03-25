import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
client = OpenAI(api_key=api_key)

class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    message: str


@app.post("/ai", response_model=MessageResponse)
def ai(body: MessageRequest):
    
    #1. enhance prompt
    #2. convert to embedding
    embedding = client.embeddings.create(
        input=body.message,
        model="text-embedding-3-small" #text-embedding-3-large
    )

    #3. retrieve top similar chunks
    #4. generate answer

    return MessageResponse(message=str(embedding.data[0].embedding))
