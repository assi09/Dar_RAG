import hashlib
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from embedder import get_embedder
from vector_store import get_client, get_vector_store

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 568M params, as specified in RAG_Pipeline.pptx
RETRIEVE_K = 50   # retrieve wide, rerank narrow — as specified in RAG_Pipeline.pptx
RERANK_TOP_N = 5  # keep only the top N after reranking


def get_reranking_retriever(embedder=None):
    """
    Two-stage retriever:
    1. Weaviate vector search fetches RETRIEVE_K candidates
    2. Cross-encoder reranker re-scores and returns top RERANK_TOP_N
    """
    if embedder is None:
        embedder = get_embedder()

    client = get_client()
    store = get_vector_store(client, embedder)
    base_retriever = store.as_retriever(search_kwargs={"k": RETRIEVE_K})

    reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=reranker, top_n=RERANK_TOP_N)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    ), client


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove chunks with identical content — caused by appendix duplicating main safeguard tables."""
    seen = set()
    unique = []
    for doc in docs:
        h = hashlib.md5(doc.page_content.strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


def retrieve_and_rerank(query: str, embedder=None) -> list[dict]:
    """Retrieve and rerank chunks for a given query."""
    retriever, client = get_reranking_retriever(embedder)
    try:
        docs: list[Document] = retriever.invoke(query)
    finally:
        client.close()

    docs = _deduplicate(docs)

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
