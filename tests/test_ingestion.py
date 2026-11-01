"""Tests for document loading, chunking, metadata, idempotency."""

import pytest

from app.ingestion.loader import (
    ALLOWED_EXTENSIONS,
    chunk_text,
    extract_source_url,
    ingest_document,
    load_document_file,
    make_chunk_id,
)


def test_chunk_text_respects_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200 + 60  # overlap slack from paragraph joining


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_prefers_paragraph_boundaries():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10)
    # Paragraphs should not be glued mid-sentence when they fit.
    assert any("First paragraph" in c for c in chunks)
    assert any("Third paragraph" in c for c in chunks)


def test_chunk_text_keeps_code_blocks_whole():
    code = "```python\n" + "\n".join(f"x = {i}" for i in range(20)) + "\n```"
    text = f"Intro paragraph.\n\n{code}\n\nOutro paragraph."
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=50)
    code_chunks = [c for c in chunks if "x = 0" in c]
    assert len(code_chunks) == 1
    # Opening and closing fences must land in the same chunk (block kept whole).
    assert "```python" in code_chunks[0]
    assert code_chunks[0].count("```") >= 2
    assert "x = 19" in code_chunks[0]


def test_make_chunk_id_stable():
    a = make_chunk_id("doc.md", 0, "hello world")
    b = make_chunk_id("doc.md", 0, "hello world")
    c = make_chunk_id("doc.md", 1, "hello world")
    assert a == b
    assert a != c


def test_extract_source_url():
    content = "# Title\n\nSource: https://fastapi.tiangolo.com/tutorial/dependencies/\n\nBody"
    assert extract_source_url(content) == "https://fastapi.tiangolo.com/tutorial/dependencies/"
    assert extract_source_url("no url here") is None


def test_ingest_document_adds_chunks_and_metadata(fake_collection):
    content = (
        "# Dependencies\n\nSource: https://fastapi.tiangolo.com/tutorial/dependencies/\n\n"
        + "Dependency injection lets endpoints declare what they need. " * 40
    )
    added = ingest_document("dependencies.md", content)
    assert added > 0
    assert len(fake_collection.store) == added
    for _id, (doc_text, meta) in fake_collection.store.items():
        assert meta["document"] == "dependencies.md"
        assert meta["chunk_id"].startswith("chunk_")
        assert meta["source_url"] == "https://fastapi.tiangolo.com/tutorial/dependencies/"


def test_ingest_document_is_idempotent(fake_collection):
    content = "Some stable content. " * 50
    first = ingest_document("intro.md", content)
    second = ingest_document("intro.md", content)
    assert first > 0
    assert second == 0
    assert len(fake_collection.store) == first


def test_ingest_document_rejects_empty(fake_collection):
    with pytest.raises(ValueError):
        ingest_document("empty.md", "   ")


def test_load_document_file_rejects_executable(tmp_path):
    evil = tmp_path / "run.exe"
    evil.write_bytes(b"MZ fake binary")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document_file(evil)


def test_load_document_file_reads_md(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("# Hello\n\nWorld", encoding="utf-8")
    name, content = load_document_file(doc)
    assert name == "notes.md"
    assert "Hello" in content


def test_allowed_extensions_excludes_executables():
    assert ".exe" not in ALLOWED_EXTENSIONS
    assert ".md" in ALLOWED_EXTENSIONS and ".txt" in ALLOWED_EXTENSIONS
