"""
Test different hybrid search alpha values against the golden dataset.
alpha=1.0 → pure vector search
alpha=0.75 → 75% vector, 25% BM25
alpha=0.5 → equal blend (Weaviate default)
alpha=0.25 → 25% vector, 75% BM25
alpha=0.0 → pure BM25
"""
import json
import sys
sys.path.insert(0, 'src')

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from embedder import get_embedder
from vector_store import get_client, get_vector_store

DATASET_PATH = "data/golden_dataset.json"
RETRIEVE_K   = 50
RERANK_TOP_N = 5
ALPHAS       = [1.0, 0.75, 0.5, 0.25, 0.0]


def retrieve_with_alpha(query: str, store, reranker_compressor, alpha: float) -> list[dict]:
    docs_and_scores = store.similarity_search_with_score(query, k=RETRIEVE_K, alpha=alpha)
    docs = [doc for doc, _ in docs_and_scores]
    reranked = reranker_compressor.compress_documents(docs, query)
    return [
        {
            "content": d.page_content,
            "section": d.metadata.get("section_title", ""),
            "page": d.metadata.get("page_number", ""),
        }
        for d in reranked
    ]


def section_hit(results, expected_pages):
    retrieved = {int(r["page"]) for r in results if r["page"]}
    return bool(retrieved & set(expected_pages))


def keyword_score(results, keywords):
    text = " ".join(r["content"].lower() for r in results)
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return round(hits / len(keywords), 2) if keywords else 0.0


def evaluate_alpha(alpha, dataset, store, reranker_compressor):
    hits = 0
    kw_total = 0.0
    for entry in dataset:
        results = retrieve_with_alpha(entry["question"], store, reranker_compressor, alpha)
        hits    += int(section_hit(results, entry["source_pages"]))
        kw_total += keyword_score(results, entry["keywords"])
    return hits, round(kw_total / len(dataset), 2)


def main():
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    print("Loading models and connecting to Weaviate...")
    embedder   = get_embedder()
    client     = get_client()
    store      = get_vector_store(client, embedder)
    reranker   = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=reranker, top_n=RERANK_TOP_N)

    print(f"\n{'Alpha':<8} {'Section Hit':<14} {'Keyword Score':<15} {'Verdict'}")
    print("-" * 55)

    best_alpha = None
    best_score = -1

    for alpha in ALPHAS:
        hits, kw = evaluate_alpha(alpha, dataset, store, compressor)
        hit_pct  = hits / len(dataset)
        combined = hit_pct * 0.6 + kw * 0.4  # weight section hit more

        verdict = ""
        if combined > best_score:
            best_score = combined
            best_alpha = alpha
            verdict = "<-- best so far"

        label = {1.0: "pure vector", 0.75: "75% vec/25% BM25",
                 0.5: "50/50 (default)", 0.25: "25% vec/75% BM25",
                 0.0: "pure BM25"}.get(alpha, "")

        print(f"  {alpha:<6} {hits}/{len(dataset)} ({hit_pct:.0%})     {kw:.0%}           {verdict}")
        print(f"         ({label})")

    print("-" * 55)
    print(f"\nBest alpha: {best_alpha}")
    client.close()


if __name__ == "__main__":
    main()
