import re

from app import config
from app.rag.retrieval import search_documents
from app.rag.state import RAGState

FALLBACK_MESSAGE = (
    "I couldn't find sufficient information in the available documentation "
    "to answer this question reliably."
)

QUERY_TYPES = ("conceptual", "how-to", "troubleshooting", "api-reference")


def get_llm():
    """Build the Gemini chat model. Raises a clear error without an API key."""
    api_key = config.require_api_key()
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Node 1: Query Analysis
# ---------------------------------------------------------------------------
ANALYZE_PROMPT = """You are helping a technical documentation search engine.
Rewrite the user's question to improve document retrieval.

User question: {question}
{retry_hint}

Do two things:
1. Rewrite the question: fix ambiguity, expand abbreviations, add relevant
   FastAPI technical terms (e.g. "dependency injection", "Pydantic model",
   "path parameter", "HTTPException").
2. Classify it as exactly one of: conceptual, how-to, troubleshooting, api-reference.

Reply in exactly this format:
REWRITTEN: <your rewritten question>
TYPE: <one of the four types>"""


def analyze_query(state: RAGState) -> dict:
    llm = get_llm()
    question = state["original_question"]
    retry_count = state.get("retry_count", 0)

    retry_hint = ""
    if retry_count > 0:
        previous = state.get("rewritten_query", question)
        retry_hint = (
            f"The previous search query was: {previous}\n"
            "It returned no relevant documents. Rewrite the question to be "
            "different and more specific — try synonyms, related FastAPI "
            "concepts, or a different phrasing."
        )

    response = llm.invoke(
        ANALYZE_PROMPT.format(question=question, retry_hint=retry_hint)
    )
    text = response.content if hasattr(response, "content") else str(response)

    rewritten = question
    query_type = "conceptual"
    rewritten_match = re.search(r"REWRITTEN:\s*(.+)", text)
    if rewritten_match:
        rewritten = rewritten_match.group(1).strip()
    type_match = re.search(r"TYPE:\s*([a-z\-]+)", text, re.IGNORECASE)
    if type_match and type_match.group(1).lower() in QUERY_TYPES:
        query_type = type_match.group(1).lower()

    return {"rewritten_query": rewritten, "query_type": query_type}


# ---------------------------------------------------------------------------
# Node 2: Retrieval
# ---------------------------------------------------------------------------
def retrieve_documents(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["original_question"]
    hits = search_documents(query, top_k=config.TOP_K)
    return {"retrieved_documents": hits}


# ---------------------------------------------------------------------------
# Node 3: Document Grading
# ---------------------------------------------------------------------------
GRADE_PROMPT = """You are grading whether a document chunk helps answer a question.

Question: {question}

Document chunk (from {document}, {chunk_id}):
---
{chunk_text}
---

Does this chunk contain information relevant to answering the question?
Reply with exactly YES or NO, then a one-line reason.
Format:
VERDICT: YES
REASON: <short reason>"""


def _grade_single_chunk(llm, question: str, chunk: dict) -> dict:
    response = llm.invoke(
        GRADE_PROMPT.format(
            question=question,
            document=chunk.get("document", "unknown"),
            chunk_id=chunk.get("chunk_id", ""),
            chunk_text=chunk.get("text", "")[:2000],
        )
    )
    text = response.content if hasattr(response, "content") else str(response)
    verdict_match = re.search(r"VERDICT:\s*(YES|NO)", text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", text)
    relevant = bool(
        verdict_match and verdict_match.group(1).upper() == "YES"
    )
    reason = reason_match.group(1).strip() if reason_match else ""
    return {
        "document": chunk.get("document", "unknown"),
        "chunk_id": chunk.get("chunk_id", ""),
        "relevant": relevant,
        "reason": reason,
    }


def grade_documents(state: RAGState) -> dict:
    llm = get_llm()
    question = state["original_question"]
    chunks = state.get("retrieved_documents", [])

    results = [_grade_single_chunk(llm, question, c) for c in chunks]
    relevant_ids = {
        (r["document"], r["chunk_id"]) for r in results if r["relevant"]
    }
    relevant = [
        c
        for c in chunks
        if (c.get("document"), c.get("chunk_id")) in relevant_ids
    ]
    return {"grading_results": results, "relevant_documents": relevant}


# ---------------------------------------------------------------------------
# Node 6: Answer Generation
# ---------------------------------------------------------------------------
ANSWER_PROMPT = """Answer the user's question using ONLY the documentation context below.
Do not use general knowledge beyond what the context supports.
If the context lacks enough information, say so explicitly.

Question: {question}

Context:
{context}

Write a clear, technically accurate answer. After each key fact, add a
citation in this exact format: [Source: <document_name>, Chunk: <chunk_id>]
Example: [Source: dependencies.md, Chunk: chunk_2]"""


def _format_context(documents: list[dict]) -> str:
    blocks = []
    for doc in documents:
        blocks.append(
            f"[{doc.get('document')}, {doc.get('chunk_id')}]\n{doc.get('text', '')}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(state: RAGState) -> dict:
    llm = get_llm()
    question = state["original_question"]
    relevant = state.get("relevant_documents", [])

    context = _format_context(relevant)
    response = llm.invoke(ANSWER_PROMPT.format(question=question, context=context))
    answer = response.content if hasattr(response, "content") else str(response)

    sources = [
        {
            "document": doc.get("document", "unknown"),
            "chunk_id": doc.get("chunk_id", ""),
            "source_url": doc.get("source_url"),
        }
        for doc in relevant
    ]
    return {"answer": answer.strip(), "sources": sources, "status": "success"}


# ---------------------------------------------------------------------------
# Node 7: Fallback
# ---------------------------------------------------------------------------
def fallback(state: RAGState) -> dict:
    return {"answer": FALLBACK_MESSAGE, "sources": [], "status": "fallback"}


# ---------------------------------------------------------------------------
# Optional: Hallucination / support check (bonus)
# ---------------------------------------------------------------------------
VERIFY_PROMPT = """You check whether an answer is supported by documentation context.

Question: {question}

Answer to check:
---
{answer}
---

Documentation context:
---
{context}
---

Is every factual claim in the answer supported by the context?
Reply in exactly this format:
VERDICT: YES
REASON: <short reason>"""


def verify_answer(state: RAGState) -> dict:
    """Bonus node: replace the answer if the LLM says it is unsupported."""
    llm = get_llm()
    relevant = state.get("relevant_documents", [])
    context = _format_context(relevant)
    response = llm.invoke(
        VERIFY_PROMPT.format(
            question=state["original_question"],
            answer=state.get("answer", ""),
            context=context,
        )
    )
    text = response.content if hasattr(response, "content") else str(response)
    verdict_match = re.search(r"VERDICT:\s*(YES|NO)", text, re.IGNORECASE)
    supported = bool(verdict_match and verdict_match.group(1).upper() == "YES")
    if supported:
        return {}
    return {
        "answer": (
            "I generated an answer, but I couldn't verify it against the "
            "retrieved documentation, so I'm not returning it. "
            + FALLBACK_MESSAGE
        ),
        "status": "fallback",
    }
