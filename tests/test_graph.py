"""Tests for LangGraph routing, retries, fallback, and node behaviour.

LLM calls are mocked with FakeLLM — no API key or network needed.
"""

from app import config
from app.rag import nodes
from app.rag.graph import _retry, build_graph, decide_next_step
from tests.conftest import FakeLLM


def _chunk(document="dependencies.md", chunk_id="chunk_0", text="dependency injection"):
    return {
        "text": text,
        "document": document,
        "chunk_id": chunk_id,
        "source_url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "distance": 0.1,
    }


# --- routing ---------------------------------------------------------------
def test_decide_next_step_routes_to_generate_when_relevant():
    state = {"relevant_documents": [_chunk()], "retry_count": 0}
    assert decide_next_step(state) == "generate"


def test_decide_next_step_retries_when_nothing_relevant():
    state = {"relevant_documents": [], "retry_count": 0}
    assert decide_next_step(state) == "retry"


def test_decide_next_step_fallback_when_retries_exhausted():
    state = {"relevant_documents": [], "retry_count": config.MAX_RETRIES}
    assert decide_next_step(state) == "fallback"


def test_retry_counter_increments():
    state = {"retry_count": 0}
    assert _retry(state)["retry_count"] == 1
    assert _retry({"retry_count": 1})["retry_count"] == 2


def test_retry_limit_prevents_infinite_loop():
    # Simulate repeated empty retrievals: counter climbs, then fallback wins.
    retry_count = 0
    steps = 0
    while True:
        state = {"relevant_documents": [], "retry_count": retry_count}
        decision = decide_next_step(state)
        steps += 1
        if decision == "retry":
            retry_count = _retry(state)["retry_count"]
        else:
            break
        assert steps < 10, "routing looped more times than the retry limit allows"
    assert decision == "fallback"
    assert retry_count == config.MAX_RETRIES


def test_graph_compiles():
    app = build_graph()
    assert app is not None


# --- fallback --------------------------------------------------------------
def test_fallback_response():
    result = nodes.fallback({})
    assert result["status"] == "fallback"
    assert "couldn't find sufficient information" in result["answer"]
    assert result["sources"] == []


# --- analyze_query ---------------------------------------------------------
def test_analyze_query_parses_rewritten_and_type(monkeypatch):
    fake = FakeLLM(["REWRITTEN: What is dependency injection in FastAPI?\nTYPE: conceptual"])
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)
    result = nodes.analyze_query({"original_question": "di in fastapi?", "retry_count": 0})
    assert result["rewritten_query"] == "What is dependency injection in FastAPI?"
    assert result["query_type"] == "conceptual"


def test_analyze_query_retry_hint_mentions_previous(monkeypatch):
    fake = FakeLLM(["REWRITTEN: new phrasing\nTYPE: how-to"])
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)
    nodes.analyze_query(
        {
            "original_question": "q?",
            "rewritten_query": "old phrasing",
            "retry_count": 1,
        }
    )
    assert "old phrasing" in fake.calls[0]


# --- grade_documents -------------------------------------------------------
def test_grade_documents_keeps_only_relevant(monkeypatch):
    chunks = [
        _chunk(chunk_id="chunk_0", text="dependency injection with Depends"),
        _chunk(chunk_id="chunk_1", text="unrelated baking recipe"),
    ]
    fake = FakeLLM(
        [
            "VERDICT: YES\nREASON: chunk explains Depends",
            "VERDICT: NO\nREASON: about baking, not FastAPI",
        ]
    )
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)
    result = nodes.grade_documents(
        {"original_question": "What is dependency injection?", "retrieved_documents": chunks}
    )
    assert len(result["grading_results"]) == 2
    assert result["grading_results"][0]["relevant"] is True
    assert result["grading_results"][1]["relevant"] is False
    assert len(result["relevant_documents"]) == 1
    assert result["relevant_documents"][0]["chunk_id"] == "chunk_0"


def test_grade_documents_all_irrelevant_gives_empty_relevant(monkeypatch):
    chunks = [_chunk(chunk_id="chunk_0", text="unrelated")]
    fake = FakeLLM(["VERDICT: NO\nREASON: not about the question"])
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)
    result = nodes.grade_documents(
        {"original_question": "Quantum physics?", "retrieved_documents": chunks}
    )
    assert result["relevant_documents"] == []
    assert decide_next_step(
        {"relevant_documents": result["relevant_documents"], "retry_count": 0}
    ) == "retry"


# --- generate_answer -------------------------------------------------------
def test_generate_answer_includes_citations_and_sources(monkeypatch):
    chunks = [
        _chunk(document="dependencies.md", chunk_id="chunk_2", text="Depends injects."),
    ]
    fake = FakeLLM(
        ["FastAPI dependency injection uses Depends. [Source: dependencies.md, Chunk: chunk_2]"]
    )
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)
    result = nodes.generate_answer(
        {"original_question": "What is dependency injection?", "relevant_documents": chunks}
    )
    assert "[Source: dependencies.md, Chunk: chunk_2]" in result["answer"]
    assert result["status"] == "success"
    assert result["sources"][0]["document"] == "dependencies.md"
    assert result["sources"][0]["chunk_id"] == "chunk_2"
    assert "fastapi.tiangolo.com" in result["sources"][0]["source_url"]


# --- verify_answer (bonus) -------------------------------------------------
def test_verify_answer_replaces_unsupported_answer(monkeypatch):
    fake = FakeLLM(["VERDICT: NO\nREASON: answer mentions things not in context"])
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)
    result = nodes.verify_answer(
        {
            "original_question": "q?",
            "answer": "FastAPI was invented on Mars.",
            "relevant_documents": [_chunk()],
        }
    )
    assert "couldn't verify" in result["answer"]


def test_verify_answer_keeps_supported_answer(monkeypatch):
    fake = FakeLLM(["VERDICT: YES\nREASON: fully supported"])
    monkeypatch.setattr(nodes, "get_llm", lambda: fake)
    result = nodes.verify_answer(
        {
            "original_question": "q?",
            "answer": "Depends injects. [Source: dependencies.md, Chunk: chunk_0]",
            "relevant_documents": [_chunk()],
        }
    )
    assert result == {}
