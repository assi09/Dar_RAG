import hashlib
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from embedder import get_embedder
from vector_store import get_client, get_vector_store

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 568M params, as specified in RAG_Pipeline.pptx
RETRIEVE_K = 50   # retrieve wide, rerank narrow — as specified in RAG_Pipeline.pptx
RERANK_TOP_N = 5  # keep only the top N after reranking


def _deduplicate(docs: list[Document]) -> list[Document]:
    """
    Remove near-duplicate chunks caused by appendix repeating main content.
    Uses section title + first 80 chars as fingerprint — handles OCR variance
    where the same content produces slightly different text across runs.
    """
    seen = set()
    unique = []
    for doc in docs:
        section = doc.metadata.get("section_title", "")
        prefix = doc.page_content.strip()[:80]
        key = hashlib.md5(f"{section}|||{prefix}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def retrieve_and_rerank(query: str, embedder=None) -> list[dict]:
    """
    Two-stage retrieval:
    1. Weaviate fetches RETRIEVE_K candidates
    2. Deduplication removes identical chunks (appendix duplicates)
    3. Cross-encoder reranker re-scores and returns top RERANK_TOP_N
    """
    if embedder is None:
        embedder = get_embedder()

    client = get_client()
    store = get_vector_store(client, embedder)

    try:
        # Step 1: broad retrieval
        docs_and_scores = store.similarity_search_with_score(query, k=RETRIEVE_K)
        docs = [doc for doc, _ in docs_and_scores]

        # Step 2: deduplicate BEFORE reranking so reranker sees unique candidates
        docs = _deduplicate(docs)

        # Step 3: rerank
        reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        compressor = CrossEncoderReranker(model=reranker, top_n=RERANK_TOP_N)
        docs = compressor.compress_documents(docs, query)
    finally:
        client.close()

    return [
        {
            "content": doc.page_content,
            "section": doc.metadata.get("section_title", ""),
            "page": doc.metadata.get("page_number", ""),
            "file": doc.metadata.get("filename", ""),
        }
        for doc in docs
    ]


if __name__ == "__main__":
    query = "What are the CIS Controls for network security?"
    print(f"Query: {query}\n")
    print("=" * 60)
    print(f"Fetching top {RETRIEVE_K} from Weaviate, reranking to top {RERANK_TOP_N}...")
    print("=" * 60)

    results = retrieve_and_rerank(query)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Page: {r['page']} | Section: {r['section'][:60]}")
        print(f"    {r['content'][:300].strip()}")
        print("-" * 60)
