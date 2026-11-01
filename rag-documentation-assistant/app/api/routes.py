"""FastAPI routes: /query, /ingest, /documents, /feedback."""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import config
from app.database import feedback as feedback_db
from app.ingestion.loader import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    fetch_url_text,
    ingest_directory,
    ingest_document,
    list_documents,
)
from app.rag.graph import answer_question

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    retry_count: int
    status: str


class IngestUrlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    document_name: str | None = Field(default=None, max_length=255)


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    rating: str = Field(pattern="^(positive|negative)$")
    comment: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse)
def query_docs(payload: QueryRequest):
    try:
        config.require_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        final_state = answer_question(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # LLM / retrieval failure -> 502, not silent
        raise HTTPException(
            status_code=502, detail=f"Failed to generate an answer: {exc}"
        )

    return QueryResponse(
        question=payload.question,
        answer=final_state.get("answer", ""),
        sources=final_state.get("sources", []),
        retry_count=final_state.get("retry_count", 0),
        status=final_state.get("status", "success"),
    )


# ---------------------------------------------------------------------------
# POST /ingest — file upload and/or URL
# ---------------------------------------------------------------------------
@router.post("/ingest")
async def ingest(
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    document_name: str | None = Form(default=None),
):
    if file is None and not url:
        raise HTTPException(
            status_code=400, detail="Provide a file upload or a document URL."
        )

    total_chunks = 0
    ingested: list[str] = []

    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )
        content_bytes = await file.read()
        if len(content_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="File too large (max 2 MB).")
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400, detail="File must be UTF-8 encoded text."
            )
        name = document_name or file.filename or "upload.md"
        try:
            added = ingest_document(name, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        total_chunks += added
        ingested.append(name)

    if url:
        try:
            text = fetch_url_text(url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Failed to fetch URL: {exc}"
            )
        name = document_name or url.rsplit("/", 1)[-1] or "url_document.txt"
        if not Path(name).suffix:
            name += ".txt"
        added = ingest_document(name, text, source_url=url)
        total_chunks += added
        ingested.append(name)

    return {
        "status": "success",
        "documents": ingested,
        "chunks_added": total_chunks,
    }


@router.post("/ingest/directory")
def ingest_local_directory():
    """Re-ingest the bundled data/documents corpus (idempotent)."""
    try:
        results = ingest_directory()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "status": "success",
        "results": results,
        "chunks_added": sum(results.values()),
    }


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------
@router.get("/documents")
def get_documents():
    try:
        documents = list_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to list documents: {exc}"
        )
    return {"status": "success", "count": len(documents), "documents": documents}


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------
@router.post("/feedback")
def submit_feedback(payload: FeedbackRequest):
    try:
        feedback_id = feedback_db.save_feedback(
            question=payload.question,
            rating=payload.rating,
            comment=payload.comment,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to store feedback: {exc}"
        )
    return {
        "status": "success",
        "message": "Thank you for your feedback.",
        "feedback_id": feedback_id,
    }
