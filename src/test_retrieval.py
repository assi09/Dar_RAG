import sys
sys.path.insert(0, 'src')

from reranker import retrieve_and_rerank

TEST_QUERIES = [
    # Specific control questions
    "What safeguards does Control 12 recommend for network infrastructure?",
    "How should organizations monitor network traffic for threats?",
    "What are the requirements for penetration testing frequency?",
    "How should enterprise assets be inventoried and tracked?",
    "What does CIS say about secure configuration of enterprise software?",
    "How should organizations manage user account privileges?",
    "What are the CIS recommendations for data recovery and backup?",
    "How should audit logs be managed and retained?",
    "What safeguards protect against malware on enterprise assets?",
    "How should vulnerability management be implemented?",
]


def run_tests():
    print("=" * 70)
    print("RETRIEVAL + RERANKING TEST SUITE")
    print("=" * 70)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Q{i}] {query}")
        print("-" * 70)
        results = retrieve_and_rerank(query)
        for j, r in enumerate(results[:3], 1):
            print(f"  ({j}) Page {r['page']:>4} | {r['section'][:55]}")
            print(f"       {r['content'][:180].strip()}")
        print()


if __name__ == "__main__":
    run_tests()
