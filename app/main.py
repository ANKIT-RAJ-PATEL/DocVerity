from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.database.feedback import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="RAG Documentation Assistant",
    description=(
        "Self-corrective RAG over FastAPI technical documentation. "
        "Ask a question, get a grounded answer with source citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(router)
