"""Shared pytest fixtures: isolate ChromaDB + feedback DB in tmp dirs."""

import os
import sys
from pathlib import Path

# Set test env BEFORE app modules are imported (config reads env at import).
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")

_TEST_TMP = Path(os.environ.get("PYTEST_TMP", "/tmp/rag_test_default"))
os.environ["CHROMA_DB_PATH"] = str(_TEST_TMP / "chroma_db")
os.environ["FEEDBACK_DB_PATH"] = str(_TEST_TMP / "feedback.db")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class FakeCollection:
    """Minimal in-memory stand-in for a ChromaDB collection."""

    def __init__(self):
        self.store = {}  # id -> (document, metadata)

    def get(self, ids=None):
        if ids is None:
            ids = list(self.store.keys())
        found = [i for i in ids if i in self.store]
        return {
            "ids": found,
            "documents": [self.store[i][0] for i in found],
            "metadatas": [self.store[i][1] for i in found],
        }

    def add(self, ids, documents, metadatas, embeddings=None):
        for i, doc, meta in zip(ids, documents, metadatas):
            self.store[i] = (doc, meta)

    def query(self, query_embeddings, n_results, include=None):
        ids = list(self.store.keys())[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.store[i][0] for i in ids]],
            "metadatas": [[self.store[i][1] for i in ids]],
            "distances": [[0.1 * (n + 1) for n in range(len(ids))]],
        }


@pytest.fixture()
def fake_collection(monkeypatch):
    from app.ingestion import loader

    class _StubEmbeddings:
        def embed_documents(self, texts):
            return [[0.0] * 16 for _ in texts]

        def embed_query(self, text):
            return [0.0] * 16

    collection = FakeCollection()
    monkeypatch.setattr(loader, "get_collection", lambda: collection)
    monkeypatch.setattr(loader, "get_embeddings", lambda: _StubEmbeddings())
    loader.reset_clients()
    return collection


@pytest.fixture()
def tmp_feedback_db(monkeypatch, tmp_path):
    from app import config

    db_path = str(tmp_path / "feedback_test.db")
    monkeypatch.setattr(config, "FEEDBACK_DB_PATH", db_path)
    yield db_path


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """LLM double returning canned responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt):
        self.calls.append(str(prompt))
        if not self._responses:
            raise AssertionError("FakeLLM ran out of canned responses")
        return FakeLLMResponse(self._responses.pop(0))
