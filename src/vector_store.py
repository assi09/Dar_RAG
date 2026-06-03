import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_core.documents import Document
from embedder import get_embedder

WEAVIATE_URL = "http://localhost:8080"
COLLECTION_NAME = "DarRAG"


def get_client() -> weaviate.WeaviateClient:
    """Connect to local Weaviate instance (must be running via Docker)."""
    return weaviate.connect_to_local(host="localhost", port=8080)


def get_vector_store(client: weaviate.WeaviateClient, embedder=None) -> WeaviateVectorStore:
    """Return a LangChain-wrapped Weaviate vector store."""
    if embedder is None:
        embedder = get_embedder()

    return WeaviateVectorStore(
        client=client,
        index_name=COLLECTION_NAME,
        text_key="text",
        embedding=embedder,
    )


def ingest(chunks: list[Document], embedder=None) -> None:
    """Embed and store chunks into Weaviate. Call after parsing + chunking."""
    client = get_client()
    try:
        store = get_vector_store(client, embedder)
        store.add_documents(chunks)
        print(f"Ingested {len(chunks)} chunks into collection '{COLLECTION_NAME}'")
    finally:
        client.close()


if __name__ == "__main__":
    print("Connecting to Weaviate at localhost:8080 ...")
    client = get_client()
    try:
        meta = client.get_meta()
        print(f"Weaviate version : {meta['version']}")
        print(f"Connection       : OK")
    finally:
        client.close()
