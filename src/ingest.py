import time
from parser import load_documents
from chunker import chunk_documents
from embedder import get_embedder
from vector_store import ingest


def run_ingestion():
    total_start = time.time()

    print("=" * 60)
    print("STEP 1/4 — Parsing documents")
    print("=" * 60)
    t = time.time()
    docs = load_documents()
    print(f"Done in {time.time() - t:.1f}s\n")

    print("=" * 60)
    print("STEP 2/4 — Chunking sections")
    print("=" * 60)
    t = time.time()
    chunks = chunk_documents(docs)
    print(f"Done in {time.time() - t:.1f}s\n")

    print("=" * 60)
    print("STEP 3/4 — Loading embedding model")
    print("=" * 60)
    t = time.time()
    embedder = get_embedder()
    print(f"Embedding model ready in {time.time() - t:.1f}s\n")

    print("=" * 60)
    print("STEP 4/4 — Embedding and storing in Weaviate")
    print("=" * 60)
    t = time.time()
    ingest(chunks, embedder)
    print(f"Done in {time.time() - t:.1f}s\n")

    print("=" * 60)
    print(f"Ingestion complete — {len(chunks)} chunks stored")
    print(f"Total time: {time.time() - total_start:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
