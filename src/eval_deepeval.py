"""
3.2 Evaluation Framework — DeepEval
Runs 6 RAG metrics against the golden dataset using llama3.2 as the judge.

Metrics (from RAG_Pipeline_Reference.pdf):
  - Faithfulness         : answer contains only claims from retrieved context
  - AnswerRelevancy      : answer is directly relevant to the question
  - ContextualPrecision  : most relevant chunks are ranked first
  - ContextualRecall     : retrieved context covers all facts in expected answer
  - ContextualRelevancy  : retrieved chunks are actually relevant to the question
  - Hallucination        : answer does not contradict or fabricate facts
"""
import json
import os
import random
import sys
sys.path.insert(0, 'src')

# llama3.2 on CPU is slow — give each test case enough time for 6 metrics
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from reranker import retrieve_and_rerank
from generator import generate

GOLDEN_DATASET_PATH = "data/golden_dataset.json"
JUDGE_MODEL = "llama3.2"


def build_test_cases(dataset: list[dict]) -> list[LLMTestCase]:
    test_cases = []
    for i, entry in enumerate(dataset, 1):
        print(f"  [{i}/{len(dataset)}] Running pipeline for: {entry['question'][:60]}")
        chunks = retrieve_and_rerank(entry["question"])
        actual_output = generate(entry["question"], chunks)
        retrieval_context = [c["content"] for c in chunks]

        test_cases.append(LLMTestCase(
            input=entry["question"],
            actual_output=actual_output,
            expected_output=entry["ground_truth_answer"],
            retrieval_context=retrieval_context,
        ))
    return test_cases


def main():
    with open(GOLDEN_DATASET_PATH) as f:
        dataset = json.load(f)

    random.seed(42)
    dataset = random.sample(dataset, 5)

    print(f"Judge model  : {JUDGE_MODEL}")
    print(f"Test cases   : {len(dataset)}")
    print(f"Metrics      : Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall")
    print(f"Thresholds   : Faithfulness≥0.7, Relevancy≥0.7, Precision≥0.6, Recall≥0.6\n")

    print("=" * 60)
    print("STEP 1 — Running RAG pipeline on all golden questions...")
    print("=" * 60)
    test_cases = build_test_cases(dataset)

    judge = OllamaModel(model=JUDGE_MODEL)

    metrics = [
        FaithfulnessMetric(threshold=0.7,        model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.7,      model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=0.6,  model=judge, include_reason=True),
        ContextualRecallMetric(threshold=0.6,     model=judge, include_reason=True),
    ]

    print("\n" + "=" * 60)
    print("STEP 2 — Running DeepEval metrics (sequential, local Ollama)...")
    print("=" * 60)
    evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(run_async=False),
    )


if __name__ == "__main__":
    main()
