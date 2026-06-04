"""
Evaluation Framework — DeepEval with Groq API judge (manual golden dataset).
Same golden dataset as eval_deepeval.py but uses groq/llama-3.3-70b-versatile
as judge for reliable, consistent scoring.

Set your key before running:
  export GROQ_API_KEY=gsk_...
  HF_HUB_DISABLE_XET=1 python3 src/eval_deepeval_groq.py
"""
import json
import os
import random
import sys
sys.path.insert(0, 'src')

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "120"

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from groq_judge import GroqJudge
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from embedder import get_embedder
from vector_store import get_client, get_vector_store
from reranker import _deduplicate, RETRIEVE_K, RERANK_TOP_N, RERANKER_MODEL
from generator import generate

GOLDEN_DATASET_PATH = "data/golden_dataset.json"


def retrieve_once(query: str, store, compressor) -> list[dict]:
    docs_and_scores = store.similarity_search_with_score(query, k=RETRIEVE_K)
    docs = _deduplicate([doc for doc, _ in docs_and_scores])
    docs = compressor.compress_documents(docs, query)
    return [
        {"content": d.page_content, "section": d.metadata.get("section_title", ""),
         "page": d.metadata.get("page_number", ""), "file": d.metadata.get("filename", "")}
        for d in docs
    ]


def build_test_cases(dataset: list[dict]) -> list[LLMTestCase]:
    print("Loading pipeline models (shared across all queries)...")
    embedder   = get_embedder()
    client     = get_client()
    store      = get_vector_store(client, embedder)
    reranker   = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=reranker, top_n=RERANK_TOP_N)

    test_cases = []
    try:
        for i, entry in enumerate(dataset, 1):
            print(f"  [{i}/{len(dataset)}] {entry['question'][:60]}")
            chunks        = retrieve_once(entry["question"], store, compressor)
            actual_output = generate(entry["question"], chunks)
            test_cases.append(LLMTestCase(
                input=entry["question"],
                actual_output=actual_output,
                expected_output=entry["ground_truth_answer"],
                retrieval_context=[c["content"] for c in chunks],
            ))
    finally:
        client.close()

    return test_cases


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: Set GROQ_API_KEY before running.")
        sys.exit(1)

    with open(GOLDEN_DATASET_PATH) as f:
        dataset = json.load(f)

    random.seed(42)
    dataset = random.sample(dataset, 5)

    judge = GroqJudge()
    print(f"Judge        : {judge.get_model_name()}")
    print(f"Dataset      : manual golden dataset (data/golden_dataset.json)")
    print(f"Test cases   : {len(dataset)}")
    print(f"Metrics      : Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall\n")

    print("=" * 60)
    print("STEP 1 — Running RAG pipeline...")
    print("=" * 60)
    test_cases = build_test_cases(dataset)

    metrics = [
        FaithfulnessMetric(threshold=0.7,       model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.7,     model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=0.6, model=judge, include_reason=True),
        ContextualRecallMetric(threshold=0.6,    model=judge, include_reason=True),
    ]

    print("\n" + "=" * 60)
    print("STEP 2 — Running DeepEval metrics (Groq judge)...")
    print("=" * 60)
    evaluate(test_cases, metrics, async_config=AsyncConfig(run_async=False))


if __name__ == "__main__":
    main()
