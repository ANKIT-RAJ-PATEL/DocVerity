import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; environment variables still work without it.
    pass


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "./data/documents")
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "fastapi_docs")
MAX_RETRIES: int = _get_int("MAX_RETRIES", 2)
TOP_K: int = _get_int("TOP_K", 5)
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 150)
FEEDBACK_DB_PATH: str = os.getenv("FEEDBACK_DB_PATH", "./feedback.db")


def require_api_key() -> str:
    """Return the Gemini API key or raise a clear error."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "Google Gemini API key to enable live inference."
        )
    return GEMINI_API_KEY
