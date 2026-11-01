"""End-to-end ingestion verification with REAL ChromaDB persistence.

Uses a tiny deterministic embedding function instead of sentence-transformers
(so no torch download is needed here); everything else is the real code path:
chunking, idempotent upsert, metadata, persistence across restarts, listing.
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("GEMINI_API_KEY", "test-key")

TMP_CHROMA = "/tmp/rag_verify_chroma"
shutil.rmtree(TMP_CHROMA, ignore_errors=True)

from app import config

config.CHROMA_DB_PATH = TMP_CHROMA

from chromadb.api.types import Documents, Embeddings


class TinyEmbeddingFunction:
    """Deterministic 16-dim hash embedding — only for verification."""

    def name(self) -> str:
        return "tiny-hash"

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([b / 255.0 for b in digest[:16]])
        return vectors


def _patched_get_collection():
    import chromadb

    os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, embedding_function=TinyEmbeddingFunction()
    )


import app.ingestion.loader as loader

loader.get_collection = _patched_get_collection

# 1. Ingest the real corpus.
results = loader.ingest_directory()
total = sum(results.values())
print("ingest results:", results)
assert total > 0, "expected chunks to be ingested"
assert set(results) == {
    "introduction.md",
    "path_parameters.md",
    "request_body.md",
    "dependencies.md",
    "error_handling.md",
}

# 2. Idempotency: second run adds nothing.
results2 = loader.ingest_directory()
assert sum(results2.values()) == 0, f"re-ingest added chunks: {results2}"
print("idempotency OK: second ingest added 0 chunks")

# 3. Metadata preserved.
docs = loader.list_documents()
print("documents:", docs)
assert len(docs) == 5
for d in docs:
    assert d["chunks"] > 0
deps = next(d for d in docs if d["document"] == "dependencies.md")
assert deps["source_url"] == "https://fastapi.tiangolo.com/tutorial/dependencies/"

# 4. Persistence across a fresh client (simulates restart).
import importlib

importlib.reload(loader)
loader.get_collection = _patched_get_collection
docs_after = loader.list_documents()
assert len(docs_after) == 5, "documents did not persist across restart"
print("persistence OK: 5 documents survive a fresh client")

# 5. Retrieval path returns metadata.
class TinyQueryEmbeddings:
    def embed_query(self, text: str):
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[:16]]

loader.get_embeddings = lambda: TinyQueryEmbeddings()
import app.rag.retrieval as retrieval

hits = retrieval.search_documents("dependency injection Depends", top_k=3)
assert len(hits) == 3
assert all("document" in h and "chunk_id" in h and "source_url" in h for h in hits)
print("retrieval OK, top hit:", hits[0]["document"], hits[0]["chunk_id"])

shutil.rmtree(TMP_CHROMA, ignore_errors=True)
print("ALL INGESTION CHECKS PASSED")
