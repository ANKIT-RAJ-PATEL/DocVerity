from typing import TypedDict


class DocumentChunk(TypedDict, total=False):
    text: str
    document: str
    chunk_id: str
    source_url: str | None
    distance: float


class GradingResult(TypedDict):
    document: str
    chunk_id: str
    relevant: bool
    reason: str


class RAGState(TypedDict, total=False):
    original_question: str
    rewritten_query: str
    query_type: str  # conceptual | how-to | troubleshooting | api-reference
    retrieved_documents: list[DocumentChunk]
    grading_results: list[GradingResult]
    relevant_documents: list[DocumentChunk]
    retry_count: int
    answer: str
    sources: list[dict]
    status: str  # success | fallback
