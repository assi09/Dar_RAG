"""
Evaluation Framework — DeepEval with Synthesizer (auto-generated golden dataset).
Uses Groq to generate Q&A pairs from Weaviate chunks, then evaluates the
RAG pipeline against those generated goldens.

This is the SYNTHESIZED dataset version — for the MANUAL dataset see:
  eval_deepeval.py      (Ollama judge)
  eval_deepeval_groq.py (Groq judge)

Set your key before running:
  export GROQ_API_KEY=gsk_...
  HF_HUB_DISABLE_XET=1 python3 src/eval_deepeval_synth.py
"""
import os
import sys
sys.path.insert(0, 'src')

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "120"

from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
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

SYNTH_OUTPUT_DIR = "data"
N_GOLDENS = 10  # number of Q&A pairs to generate


def generate_synthetic_goldens(judge: GroqJudge) -> EvaluationDataset:
    """Pull chunks from Weaviate and use Synthesizer to generate Q&A pairs."""
    print("Fetching chunks from Weaviate as context source...")
    embedder = get_embedder()
    client   = get_client()
    store    = get_vector_store(client, embedder)

    # Diverse queries to get varied chunk samples
    queries = [
        "network infrastructure security safeguards",
        "vulnerability management patch management",
        "audit log management retention",
        "malware defense anti-exploitation",
        "account management privileges MFA",
    ]
    seen, contexts = set(), []
    for q in queries:
        results = store.similarity_search(q, k=4)
        for doc in results:
            key = doc.page_content[:80]
            if key not in seen and len(doc.page_content.strip()) > 100:
                seen.add(key)
                contexts.append([doc.page_content])
            if len(contexts) >= N_GOLDENS:
                break
        if len(contexts) >= N_GOLDENS:
            break
    client.close()

    print(f"Generating {len(contexts)} synthetic goldens using {judge.get_model_name()}...")
    synthesizer = Synthesizer(model=judge)
    synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
    )

    dataset = EvaluationDataset(goldens=synthesizer.synthetic_goldens)
    dataset.save_as(file_type="json", directory=SYNTH_OUTPUT_DIR)
    print(f"Saved synthetic goldens to {SYNTH_OUTPUT_DIR}/\n")
    return dataset


def retrieve_once(query: str, store, compressor) -> list[dict]:
    docs_and_scores = store.similarity_search_with_score(query, k=RETRIEVE_K)
    docs = _deduplicate([doc for doc, _ in docs_and_scores])
    docs = compressor.compress_documents(docs, query)
    return [
        {"content": d.page_content, "section": d.metadata.get("section_title", ""),
         "page": d.metadata.get("page_number", ""), "file": d.metadata.get("filename", "")}
        for d in docs
    ]


def build_test_cases(dataset: EvaluationDataset) -> list[LLMTestCase]:
    print("Loading pipeline models (shared across all queries)...")
    embedder   = get_embedder()
    client     = get_client()
    store      = get_vector_store(client, embedder)
    reranker   = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=reranker, top_n=RERANK_TOP_N)

    test_cases = []
    try:
        for i, golden in enumerate(dataset.goldens, 1):
            print(f"  [{i}/{len(dataset.goldens)}] {golden.input[:60]}")
            chunks        = retrieve_once(golden.input, store, compressor)
            actual_output = generate(golden.input, chunks)
            test_cases.append(LLMTestCase(
                input=golden.input,
                actual_output=actual_output,
                expected_output=golden.expected_output,
                retrieval_context=[c["content"] for c in chunks],
            ))
    finally:
        client.close()

    return test_cases


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: Set GROQ_API_KEY before running.")
        sys.exit(1)

    judge = GroqJudge()
    print(f"Judge        : {judge.get_model_name()}")
    print(f"Dataset      : auto-generated via Synthesizer")
    print(f"Goldens      : {N_GOLDENS}\n")

    print("=" * 60)
    print("STEP 1 — Generating synthetic golden dataset...")
    print("=" * 60)
    dataset = generate_synthetic_goldens(judge)

    print("=" * 60)
    print("STEP 2 — Running RAG pipeline on generated questions...")
    print("=" * 60)
    test_cases = build_test_cases(dataset)

    metrics = [
        FaithfulnessMetric(threshold=0.7,       model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.7,     model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=0.6, model=judge, include_reason=True),
        ContextualRecallMetric(threshold=0.6,    model=judge, include_reason=True),
    ]

    print("\n" + "=" * 60)
    print("STEP 3 — Running DeepEval metrics...")
    print("=" * 60)
    evaluate(test_cases, metrics, async_config=AsyncConfig(run_async=False))


if __name__ == "__main__":
    main()
