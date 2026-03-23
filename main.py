from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    message: str


@app.post("/ai", response_model=MessageResponse)
def ai(body: MessageRequest):
    return MessageResponse(message=body.message)
