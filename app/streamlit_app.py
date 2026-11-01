import streamlit as st

from app import config
from app.database import feedback as feedback_db
from app.ingestion.loader import (
    ALLOWED_EXTENSIONS,
    fetch_url_text,
    ingest_directory,
    ingest_document,
    list_documents,
)
from app.rag.graph import answer_question

st.set_page_config(page_title="RAG Documentation Assistant", layout="wide")
st.title("RAG Documentation Assistant")
st.caption("Self-corrective RAG over FastAPI technical documentation")

tab_ask, tab_ingest, tab_docs, tab_feedback = st.tabs(
    ["Ask", "Ingest", "Documents", "Feedback"]
)

# ---------------------------------------------------------------------------
# Tab 1: Ask
# ---------------------------------------------------------------------------
with tab_ask:
    st.subheader("Ask a question")
    question = st.text_area(
        "Your question",
        placeholder="What is dependency injection in FastAPI?",
        height=80,
        key="question_input",
    )
    if st.button("Get answer", type="primary", key="ask_button"):
        if not question.strip():
            st.warning("Please type a question first.")
        elif not config.GEMINI_API_KEY:
            st.error(
                "GEMINI_API_KEY is not set. Add it to your .env file and restart "
                "the app to enable live answers."
            )
        else:
            with st.spinner("Running the RAG workflow..."):
                try:
                    result = answer_question(question.strip())
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")
                    result = None
            if result:
                st.session_state["last_result"] = result
                st.session_state["last_question"] = question.strip()

    result = st.session_state.get("last_result")
    if result:
        status = result.get("status", "")
        if status == "success":
            st.success("Answer found")
        else:
            st.warning("No reliable answer found in the documentation")
        st.markdown("### Answer")
        st.write(result.get("answer", ""))
        st.markdown("### Sources")
        sources = result.get("sources", [])
        if sources:
            for src in sources:
                url = src.get("source_url")
                label = f"{src.get('document')} ({src.get('chunk_id')})"
                if url:
                    st.markdown(f"- [{label}]({url})")
                else:
                    st.markdown(f"- {label}")
        else:
            st.caption("No sources.")
        st.caption(
            f"Retries used: {result.get('retry_count', 0)} | "
            f"Query type: {result.get('query_type', 'n/a')}"
        )

# ---------------------------------------------------------------------------
# Tab 2: Ingest
# ---------------------------------------------------------------------------
with tab_ingest:
    st.subheader("Add documents")

    st.markdown("#### Upload a file")
    uploaded = st.file_uploader(
        "Markdown or TXT file (max 2 MB)",
        type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
        key="file_uploader",
    )
    if uploaded is not None and st.button("Ingest file", key="ingest_file_btn"):
        try:
            content = uploaded.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("File must be UTF-8 encoded text.")
            content = None
        if content is not None:
            with st.spinner("Chunking and embedding..."):
                try:
                    added = ingest_document(uploaded.name, content)
                    st.success(f"Ingested {uploaded.name}: {added} new chunk(s).")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    st.markdown("#### Ingest from URL")
    url = st.text_input("Document URL", placeholder="https://fastapi.tiangolo.com/...", key="url_input")
    url_name = st.text_input(
        "Document name (optional)", placeholder="testing.md", key="url_name_input"
    )
    if st.button("Ingest URL", key="ingest_url_btn"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching and ingesting..."):
                try:
                    text = fetch_url_text(url.strip())
                    name = url_name.strip() or url.strip().rsplit("/", 1)[-1] or "url_document.txt"
                    if "." not in name:
                        name += ".txt"
                    added = ingest_document(name, text, source_url=url.strip())
                    st.success(f"Ingested {name}: {added} new chunk(s).")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    st.markdown("#### Bundled corpus")
    if st.button("Re-ingest bundled FastAPI docs", key="ingest_corpus_btn"):
        with st.spinner("Ingesting..."):
            try:
                results = ingest_directory()
                total = sum(results.values())
                st.success(f"Done: {total} new chunk(s) across {len(results)} documents.")
                st.json(results)
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

# ---------------------------------------------------------------------------
# Tab 3: Documents
# ---------------------------------------------------------------------------
with tab_docs:
    st.subheader("Indexed documents")
    if st.button("Refresh", key="docs_refresh_btn"):
        st.rerun()
    try:
        documents = list_documents()
    except Exception as exc:
        st.error(f"Could not list documents: {exc}")
        documents = []
    if documents:
        st.dataframe(
            [
                {
                    "Document": d["document"],
                    "Source URL": d.get("source_url") or "-",
                    "Chunks": d["chunks"],
                }
                for d in documents
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No documents indexed yet. Use the Ingest tab.")

# ---------------------------------------------------------------------------
# Tab 4: Feedback
# ---------------------------------------------------------------------------
with tab_feedback:
    st.subheader("Rate an answer")
    default_question = st.session_state.get("last_question", "")
    fb_question = st.text_input("Question", value=default_question, key="fb_question")
    fb_rating = st.radio("Rating", ["positive", "negative"], horizontal=True, key="fb_rating")
    fb_comment = st.text_area("Comment (optional)", key="fb_comment")
    if st.button("Submit feedback", type="primary", key="fb_submit_btn"):
        if not fb_question.strip():
            st.warning("Please enter the question this feedback is about.")
        else:
            try:
                fb_id = feedback_db.save_feedback(
                    question=fb_question.strip(),
                    rating=fb_rating,
                    comment=fb_comment.strip() or None,
                )
                st.success(f"Thank you! Feedback saved (id {fb_id}).")
            except Exception as exc:
                st.error(f"Could not save feedback: {exc}")

    st.markdown("#### Recent feedback")
    try:
        recent = feedback_db.list_feedback(limit=20)
    except Exception:
        recent = []
    if recent:
        st.dataframe(
            [
                {
                    "Question": r["question"],
                    "Rating": r["rating"],
                    "Comment": r.get("comment") or "-",
                    "At": r["created_at"],
                }
                for r in recent
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No feedback yet.")
