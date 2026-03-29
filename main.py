from fastapi import FastAPI
from routes import embedding, similarity, ai

app = FastAPI()

app.include_router(embedding.router)
app.include_router(similarity.router)
app.include_router(ai.router)
