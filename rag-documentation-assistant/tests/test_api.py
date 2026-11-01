"""API tests — the /query LLM path is mocked; validation is real."""

import io

import pytest
from fastapi.testclient import TestClient

from app import config
from app.api import routes
from app.database import feedback as feedback_db
from app.main import app

client = TestClient(app)


@pytest.fixture()
def mock_answer(monkeypatch):
    state = {
        "answer": "Dependency injection uses Depends. [Source: dependencies.md, Chunk: chunk_1]",
        "sources": [
            {
                "document": "dependencies.md",
                "chunk_id": "chunk_1",
                "source_url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
            }
        ],
        "retry_count": 0,
        "status": "success",
    }
    monkeypatch.setattr(routes, "answer_question", lambda question: state)
    return state


def test_query_success(mock_answer):
    response = client.post("/query", json={"question": "What is dependency injection?"})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is dependency injection?"
    assert "Depends" in body["answer"]
    assert body["sources"][0]["document"] == "dependencies.md"
    assert body["retry_count"] == 0
    assert body["status"] == "success"


def test_query_validation_rejects_empty():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_requires_api_key(monkeypatch, mock_answer):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    response = client.post("/query", json={"question": "hello?"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_get_documents(monkeypatch):
    docs = [
        {"document": "dependencies.md", "source_url": "https://fastapi.tiangolo.com/tutorial/dependencies/", "chunks": 4}
    ]
    monkeypatch.setattr(routes, "list_documents", lambda: docs)
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["documents"][0]["chunks"] == 4


def test_feedback_storage_and_confirmation(tmp_feedback_db):
    payload = {
        "question": "What is dependency injection?",
        "rating": "positive",
        "comment": "Helpful answer.",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["feedback_id"] == 1

    stored = feedback_db.list_feedback()
    assert len(stored) == 1
    assert stored[0]["rating"] == "positive"
    assert stored[0]["comment"] == "Helpful answer."


def test_feedback_rejects_bad_rating():
    response = client.post(
        "/feedback",
        json={"question": "q?", "rating": "meh", "comment": None},
    )
    assert response.status_code == 422


def test_ingest_rejects_executable():
    files = {"file": ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream")}
    response = client.post("/ingest", files=files)
    assert response.status_code == 400


def test_ingest_requires_file_or_url():
    response = client.post("/ingest")
    assert response.status_code == 400


def test_ingest_file_upload(monkeypatch):
    monkeypatch.setattr(routes, "ingest_document", lambda name, content, source_url=None: 3)
    files = {"file": ("notes.md", io.BytesIO(b"# Title\n\nSome content"), "text/markdown")}
    response = client.post("/ingest", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["chunks_added"] == 3
    assert "notes.md" in body["documents"]
