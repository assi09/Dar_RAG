from embedder import get_embedder
from vector_store import get_client, get_vector_store

TOP_K = 5  # number of chunks to retrieve per query


def get_retriever(embedder=None):
    """Return a LangChain retriever backed by Weaviate."""
    if embedder is None:
        embedder = get_embedder()
    client = get_client()
    store = get_vector_store(client, embedder)
    return store.as_retriever(search_kwargs={"k": TOP_K}), client


def retrieve(query: str, embedder=None) -> list[dict]:
    """
    Embed the query and return the top-k most relevant chunks from Weaviate.
    Each result includes the content and its source metadata.
    """
    if embedder is None:
        embedder = get_embedder()

    client = get_client()
    try:
        store = get_vector_store(client, embedder)
        results = store.similarity_search_with_score(query, k=TOP_K)
    finally:
        client.close()

    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "score": round(score, 4),
            "section": doc.metadata.get("section_title", ""),
            "page": doc.metadata.get("page_number", ""),
            "file": doc.metadata.get("filename", ""),
        })
    return output


if __name__ == "__main__":
    query = "What are the CIS Controls for network security?"
    print(f"Query: {query}\n")
    print("=" * 60)

    results = retrieve(query)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']} | Page: {r['page']} | Section: {r['section'][:50]}")
        print(f"    {r['content'][:300].strip()}")
        print("-" * 60)
