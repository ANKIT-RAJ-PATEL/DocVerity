import hashlib
import os
import re
from pathlib import Path

from app import config

ALLOWED_EXTENSIONS = {".md", ".txt"}
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB per uploaded file

# Module-level singletons so we don't rebuild clients on every call.
_chroma_client = None
_collection = None
_embeddings = None


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Code blocks (``` fenced) are kept whole when they fit inside chunk_size;
    otherwise they fall back to plain character splitting.
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs first, but keep fenced code blocks together.
    parts = _split_paragraphs_preserving_code_blocks(text)

    chunks: list[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # If a single part is bigger than chunk_size, split it by characters.
        if len(part) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(part), chunk_size - chunk_overlap):
                chunks.append(part[i : i + chunk_size].strip())
            continue
        candidate = f"{current}\n\n{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            # Start the next chunk with overlap from the previous one.
            overlap = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap}\n\n{part}" if overlap else part

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def _split_paragraphs_preserving_code_blocks(text: str) -> list[str]:
    """Split on blank lines, but never split inside ``` fenced blocks."""
    lines = text.split("\n")
    parts: list[str] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        block = "\n".join(buf).strip()
        if block:
            parts.append(block)
        buf.clear()

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            if not in_code:
                flush()
            continue
        if not in_code and not line.strip():
            flush()
            continue
        buf.append(line)
    flush()
    return parts


def get_embeddings():
    """Lazily build the local HuggingFace embedding model (no paid API)."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def get_chroma_client():
    """Persistent ChromaDB client stored at CHROMA_DB_PATH."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return _chroma_client


def get_collection():
    """Get (or create) the ChromaDB collection for our documents.

    No embedding function is attached to the collection: embeddings are
    always computed locally via get_embeddings() and passed explicitly,
    so nothing ever needs a remote embedding API key.
    """
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(name=config.COLLECTION_NAME)
    return _collection


def reset_clients() -> None:
    """Clear cached clients (used by tests to point at temp directories)."""
    global _chroma_client, _collection, _embeddings
    _chroma_client = None
    _collection = None
    _embeddings = None


def make_chunk_id(document_name: str, chunk_index: int, chunk_text: str) -> str:
    """Stable chunk id so re-ingesting the same document is idempotent."""
    digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
    return f"{document_name}::chunk_{chunk_index}::{digest}"


def load_document_file(path: str | Path) -> tuple[str, str]:
    """Read a .md/.txt file. Returns (document_name, content)."""
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File too large (max {MAX_FILE_SIZE_BYTES} bytes)")
    content = path.read_text(encoding="utf-8")
    return path.name, content


def extract_source_url(content: str) -> str | None:
    """Pick up a 'Source: <url>' line from the top of our corpus files."""
    match = re.search(r"^Source:\s*(https?://\S+)", content, re.MULTILINE)
    return match.group(1) if match else None


def ingest_document(
    document_name: str,
    content: str,
    source_url: str | None = None,
) -> int:
    """Chunk + embed + upsert one document. Returns number of new chunks added."""
    if not content or not content.strip():
        raise ValueError("Document content is empty")

    source_url = source_url or extract_source_url(content)
    chunks = chunk_text(content)
    if not chunks:
        return 0

    collection = get_collection()
    ids = [make_chunk_id(document_name, i, c) for i, c in enumerate(chunks)]

    # Idempotency: skip chunks that already exist.
    existing = set(collection.get(ids=ids).get("ids", []))
    new_items = [
        (i, c) for i, c in zip(ids, chunks) if i not in existing
    ]
    if not new_items:
        return 0

    new_ids = [i for i, _ in new_items]
    new_chunks = [c for _, c in new_items]
    # chunk_id reflects the chunk's position among ALL chunks of the document.
    position = {cid: n for n, cid in enumerate(ids)}
    metadatas = [
        {
            "document": document_name,
            "chunk_id": f"chunk_{position[cid]}",
            "source_url": source_url or "",
        }
        for cid in new_ids
    ]

    # Embed locally (no remote API) and pass vectors explicitly.
    vectors = get_embeddings().embed_documents(new_chunks)
    collection.add(
        ids=new_ids, documents=new_chunks, metadatas=metadatas, embeddings=vectors
    )
    return len(new_ids)


def ingest_directory(documents_dir: str | Path = config.DOCUMENTS_DIR) -> dict:
    """Ingest every .md/.txt file in a directory. Returns per-file counts."""
    documents_dir = Path(documents_dir)
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    results: dict[str, int] = {}
    for path in sorted(documents_dir.iterdir()):
        if path.suffix.lower() not in ALLOWED_EXTENSIONS or not path.is_file():
            continue
        document_name, content = load_document_file(path)
        results[document_name] = ingest_document(document_name, content)
    return results


def fetch_url_text(url: str, timeout: int = 20) -> str:
    """Download a URL and return its text (HTML stripped to readable text)."""
    import httpx

    if not re.match(r"^https?://", url):
        raise ValueError("URL must start with http:// or https://")
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    html = response.text
    # Crude but dependency-free HTML -> text conversion.
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def list_documents() -> list[dict]:
    """Summarise what's indexed: one row per document name."""
    collection = get_collection()
    data = collection.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for meta in data.get("metadatas", []):
        name = meta.get("document", "unknown")
        entry = docs.setdefault(
            name, {"document": name, "source_url": meta.get("source_url") or None, "chunks": 0}
        )
        entry["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["document"])
