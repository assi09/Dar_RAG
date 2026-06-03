import sys
sys.path.insert(0, 'src')

from reranker import retrieve_and_rerank
from generator import generate


def ask(query: str) -> str:
    """Full RAG pipeline: query → retrieve → rerank → generate → answer."""
    chunks = retrieve_and_rerank(query)
    answer = generate(query, chunks)
    return answer


def interactive():
    print("=" * 60)
    print("DAR RAG — CIS Controls v8")
    print("Type your question. Press Ctrl+C to exit.")
    print("=" * 60)

    while True:
        try:
            query = input("\nQuestion: ").strip()
            if not query:
                continue
            print("\nSearching...\n")
            answer = ask(query)
            print(f"Answer:\n{answer}")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # single query mode: python3 src/main.py "your question"
        query = " ".join(sys.argv[1:])
        print(f"Q: {query}\n")
        print(ask(query))
    else:
        interactive()
