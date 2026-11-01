from langgraph.graph import END, StateGraph

from app import config
from app.rag import nodes
from app.rag.state import RAGState

ENABLE_VERIFICATION = False  # bonus hallucination check, off by default


def decide_next_step(state: RAGState) -> str:
    """Conditional routing after grading.

    - Relevant docs found  -> generate the answer.
    - None found, retries left -> bump retry_count, go back to analyze_query.
    - None found, retries exhausted -> fallback.
    """
    relevant = state.get("relevant_documents", [])
    if relevant:
        return "generate"

    retry_count = state.get("retry_count", 0)
    if retry_count < config.MAX_RETRIES:
        return "retry"
    return "fallback"


def _retry(state: RAGState) -> dict:
    # Increment here so analyze_query sees the updated count on the next pass.
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph(enable_verification: bool = ENABLE_VERIFICATION):
    graph = StateGraph(RAGState)

    graph.add_node("analyze_query", nodes.analyze_query)
    graph.add_node("retrieve_documents", nodes.retrieve_documents)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("fallback", nodes.fallback)
    graph.add_node("increment_retry", _retry)
    if enable_verification:
        graph.add_node("verify_answer", nodes.verify_answer)

    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "generate": "generate_answer",
            "retry": "increment_retry",
            "fallback": "fallback",
        },
    )
    graph.add_edge("increment_retry", "analyze_query")

    if enable_verification:
        graph.add_edge("generate_answer", "verify_answer")
        graph.add_edge("verify_answer", END)
    else:
        graph.add_edge("generate_answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def answer_question(question: str, enable_verification: bool = ENABLE_VERIFICATION) -> dict:
    """Run the full workflow for one question. Returns the final state."""
    if not question or not question.strip():
        raise ValueError("Question must not be empty")
    app = build_graph(enable_verification=enable_verification)
    initial_state: RAGState = {
        "original_question": question.strip(),
        "rewritten_query": "",
        "query_type": "",
        "retrieved_documents": [],
        "grading_results": [],
        "relevant_documents": [],
        "retry_count": 0,
        "answer": "",
        "sources": [],
        "status": "",
    }
    return app.invoke(initial_state)
