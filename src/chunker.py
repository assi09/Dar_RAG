from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHUNK_SIZE = 1400       # ~350 tokens — sweet spot for bge-small-en-v1.5 (512 token max)
CHUNK_OVERLAP = 140    # 10% overlap to preserve context across chunk boundaries
MIN_CHUNK_LENGTH = 50  # drop chunks too short to carry meaningful content


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]
    print(f"Chunked {len(docs)} sections -> {before} chunks -> {len(chunks)} after filtering noise")
    return chunks


if __name__ == "__main__":
    from parser import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)

    print("\n--- Sample Chunk ---")
    c = chunks[0]
    print(f"Source  : {c.metadata.get('filename', '')}")
    print(f"Content : {c.page_content[:300]}")
    print(f"Chars   : {len(c.page_content)}")
