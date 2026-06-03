import json
import sys
sys.path.insert(0, 'src')

from reranker import retrieve_and_rerank

DATASET_PATH = "data/golden_dataset.json"


def keyword_hit_score(retrieved_chunks: list[dict], keywords: list[str]) -> float:
    """Score 0-1: fraction of keywords found in any retrieved chunk."""
    all_text = " ".join(r["content"].lower() for r in retrieved_chunks)
    hits = sum(1 for kw in keywords if kw.lower() in all_text)
    return round(hits / len(keywords), 2) if keywords else 0.0


def section_hit(retrieved_chunks: list[dict], expected_pages: list[int]) -> bool:
    """True if any retrieved chunk comes from an expected source page."""
    retrieved_pages = {int(r["page"]) for r in retrieved_chunks if r["page"]}
    return bool(retrieved_pages & set(expected_pages))


def evaluate():
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    print("=" * 70)
    print("GOLDEN DATASET EVALUATION")
    print(f"Dataset: {len(dataset)} questions")
    print("=" * 70)

    total_section_hits = 0
    total_keyword_score = 0.0

    for entry in dataset:
        results = retrieve_and_rerank(entry["question"])

        hit = section_hit(results, entry["source_pages"])
        kw_score = keyword_hit_score(results, entry["keywords"])

        total_section_hits += int(hit)
        total_keyword_score += kw_score

        status = "PASS" if hit else "FAIL"
        print(f"\n[Q{entry['id']}] {status} | Keywords: {kw_score:.0%} | {entry['question'][:60]}")
        print(f"  Expected pages : {entry['source_pages']}")
        print(f"  Retrieved pages: {sorted({int(r['page']) for r in results if r['page']})}")
        if not hit:
            print(f"  Top section    : {results[0]['section'][:60] if results else 'none'}")

    print("\n" + "=" * 70)
    print(f"Section Hit Rate : {total_section_hits}/{len(dataset)} ({total_section_hits/len(dataset):.0%})")
    print(f"Avg Keyword Score: {total_keyword_score/len(dataset):.0%}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
