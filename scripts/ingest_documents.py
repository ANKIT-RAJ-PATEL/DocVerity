"""Ingest the bundled FastAPI documentation corpus into ChromaDB.

Usage:
    python scripts/ingest_documents.py
"""

import sys
from pathlib import Path

# Allow running as `python scripts/ingest_documents.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.loader import ingest_directory


def main() -> None:
    results = ingest_directory()
    total = sum(results.values())
    print("Ingestion complete:")
    for name, count in results.items():
        print(f"  {name}: {count} new chunk(s)")
    print(f"Total new chunks: {total}")


if __name__ == "__main__":
    main()
