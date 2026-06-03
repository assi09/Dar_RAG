from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embedder() -> HuggingFaceEmbeddings:
    """Return the embedding model. Downloads once, cached permanently."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # required for bge models to work correctly
    )


if __name__ == "__main__":
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    embedder = get_embedder()

    test_sentences = [
        "What are the CIS Critical Security Controls?",
        "How do you configure network access control?",
    ]
    vectors = embedder.embed_documents(test_sentences)

    print(f"\nModel loaded successfully.")
    print(f"Embedding dimensions : {len(vectors[0])}")
    print(f"Sample vector (first 5 values): {vectors[0][:5]}")
