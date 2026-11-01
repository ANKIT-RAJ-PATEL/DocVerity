from app import config
from app.ingestion.loader import get_collection, get_embeddings


def search_documents(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Embed the query locally and return the top_k chunks with metadata.

    Each result: {"text", "document", "chunk_id", "source_url", "distance"}.
    Lower distance = more similar.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    hits = []
    for text, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        hits.append(
            {
                "text": text,
                "document": meta.get("document", "unknown"),
                "chunk_id": meta.get("chunk_id", ""),
                "source_url": meta.get("source_url") or None,
                "distance": float(distance),
            }
        )
    return hits
