from collections import Counter
from parser import load_documents
from chunker import chunk_documents


def verify():
    print("=" * 60)
    print("STAGE 2.1 — PARSING VERIFICATION")
    print("=" * 60)
    docs = load_documents()

    # Element type breakdown
    types = Counter(d.metadata.get("category", "unknown") for d in docs)
    print("\nElement types found (paged mode — will mostly be UncategorizedText):")
    for t, count in types.most_common():
        print(f"  {t:<25} {count}")

    # Check metadata fields
    sample = docs[10]
    print(f"\nSample element metadata: {sample.metadata}")
    print(f"Sample content        : {sample.page_content[:200]}")

    print("\n" + "=" * 60)
    print("STAGE 2.2 — CHUNKING VERIFICATION")
    print("=" * 60)
    chunks = chunk_documents(docs)

    # Chunk size distribution
    sizes = [len(c.page_content) for c in chunks]
    print(f"\nChunk size stats:")
    print(f"  Min   : {min(sizes)} chars")
    print(f"  Max   : {max(sizes)} chars")
    print(f"  Avg   : {sum(sizes) // len(sizes)} chars")
    print(f"  Over 1400 chars (bad): {sum(1 for s in sizes if s > 1400)}")

    # Show 3 varied samples
    print("\n3 sample chunks:")
    for i, idx in enumerate([0, len(chunks) // 2, len(chunks) - 1]):
        c = chunks[idx]
        print(f"\n  [{i+1}] chars={len(c.page_content)} | section={c.metadata.get('section_title','?')[:40]} | page={c.metadata.get('page_number','?')}")
        print(f"       {c.page_content[:150].strip()!r}")


if __name__ == "__main__":
    verify()
