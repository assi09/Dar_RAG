"""
2.5 Sync — file-hash gated ingestion with LangChain Indexing API
Skips files whose content hasn't changed, updates changed ones, removes deleted ones.
Run this instead of ingest.py when source documents may change over time.
"""
import hashlib
import json
from pathlib import Path

from langchain_classic.indexes import index, SQLRecordManager

from parser import load_documents, DOCS_DIR
from chunker import chunk_documents
from embedder import get_embedder
from vector_store import get_client, get_vector_store

RECORD_DB_PATH   = "data/.record_manager.db"
FILE_HASHES_PATH = "data/.file_hashes.json"
NAMESPACE        = "weaviate/DarRAG"


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_hashes() -> dict:
    p = Path(FILE_HASHES_PATH)
    return json.loads(p.read_text()) if p.exists() else {}


def save_hashes(hashes: dict):
    Path(FILE_HASHES_PATH).write_text(json.dumps(hashes, indent=2))


def get_changed_files(docs_dir: Path = DOCS_DIR) -> tuple[list[Path], list[Path]]:
    """Return (changed_files, all_current_files)."""
    supported = {".pdf", ".txt", ".docx", ".md"}
    current = [f for f in docs_dir.glob("**/*") if f.is_file() and f.suffix in supported]
    stored  = load_hashes()
    changed = [f for f in current if stored.get(str(f)) != file_md5(f)]
    return changed, current


def sync_documents():
    changed_files, all_files = get_changed_files()

    if not changed_files:
        print("All files unchanged — nothing to sync.")
        return

    print(f"{len(changed_files)} file(s) changed: {[f.name for f in changed_files]}")

    print("Loading embedding model...")
    embedder = get_embedder()

    print("Connecting to Weaviate...")
    client = get_client()
    store  = get_vector_store(client, embedder)

    print("Parsing and chunking changed documents...")
    docs   = load_documents(docs_dir=DOCS_DIR)
    chunks = chunk_documents(docs)

    record_manager = SQLRecordManager(
        namespace=NAMESPACE,
        db_url=f"sqlite:///{RECORD_DB_PATH}",
    )
    record_manager.create_schema()

    print(f"Syncing {len(chunks)} chunks...")
    result = index(
        chunks,
        record_manager,
        store,
        cleanup="incremental",
        source_id_key="source",
    )

    # persist current file hashes only after successful sync
    save_hashes({str(f): file_md5(f) for f in all_files})

    print(f"\nSync complete:")
    print(f"  Added   : {result['num_added']}")
    print(f"  Updated : {result['num_updated']}")
    print(f"  Skipped : {result['num_skipped']}")
    print(f"  Deleted : {result['num_deleted']}")

    client.close()


if __name__ == "__main__":
    sync_documents()
