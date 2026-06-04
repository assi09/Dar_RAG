"""
Pytest integration for RAG evaluation — as specified in RAG_Pipeline_Reference.pdf.

Run with:
  export GROQ_API_KEY=gsk_...
  deepeval test run src/test_rag.py

Or standard pytest:
  HF_HUB_DISABLE_XET=1 pytest src/test_rag.py -v
"""
import json
import os
import random
import sys
sys.path.insert(0, 'src')

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

from groq_judge import GroqJudge
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from embedder import get_embedder
from vector_store import get_client, get_vector_store
from reranker import _deduplicate, RETRIEVE_K, RERANK_TOP_N, RERANKER_MODEL
from generator import generate

GOLDEN_DATASET_PATH = "data/golden_dataset.json"
N_SAMPLES = 5
RANDOM_SEED = 42


def _build_pipeline():
    """Load shared models once for all test cases."""
    embedder = get_embedder()
    client = get_client()
    store = get_vector_store(client, embedder)
    reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=reranker, top_n=RERANK_TOP_N)
    return store, compressor, client


def _retrieve(query: str, store, compressor) -> list[dict]:
    docs_and_scores = store.similarity_search_with_score(query, k=RETRIEVE_K)
    docs = _deduplicate([doc for doc, _ in docs_and_scores])
    docs = compressor.compress_documents(docs, query)
    return [
        {"content": d.page_content, "section": d.metadata.get("section_title", ""),
         "page": d.metadata.get("page_number", ""), "file": d.metadata.get("filename", "")}
        for d in docs
    ]


def _load_test_cases() -> list[LLMTestCase]:
    with open(GOLDEN_DATASET_PATH) as f:
        dataset = json.load(f)

    random.seed(RANDOM_SEED)
    dataset = random.sample(dataset, min(N_SAMPLES, len(dataset)))

    store, compressor, client = _build_pipeline()
    test_cases = []
    try:
        for entry in dataset:
            chunks = _retrieve(entry["question"], store, compressor)
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


# Build test cases once at module level
_test_cases = _load_test_cases()
_judge = GroqJudge()


@pytest.mark.parametrize("test_case", _test_cases)
def test_faithfulness(test_case: LLMTestCase):
    assert_test(test_case, [FaithfulnessMetric(threshold=0.7, model=_judge)])


@pytest.mark.parametrize("test_case", _test_cases)
def test_answer_relevancy(test_case: LLMTestCase):
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7, model=_judge)])


@pytest.mark.parametrize("test_case", _test_cases)
def test_contextual_precision(test_case: LLMTestCase):
    assert_test(test_case, [ContextualPrecisionMetric(threshold=0.6, model=_judge)])


@pytest.mark.parametrize("test_case", _test_cases)
def test_contextual_recall(test_case: LLMTestCase):
    assert_test(test_case, [ContextualRecallMetric(threshold=0.6, model=_judge)])


@pytest.mark.parametrize("test_case", _test_cases)
def test_contextual_relevancy(test_case: LLMTestCase):
    assert_test(test_case, [ContextualRelevancyMetric(threshold=0.6, model=_judge)])
