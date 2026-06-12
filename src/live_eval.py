"""Live DeepEval metrics for 'rag'-classified queries.

Unlike eval_deepeval*.py (which run offline against a golden dataset with
known expected answers), these metrics require no ground truth — they score
the actual answer against the question and the retrieved context only, so
they can run on every real user query.
"""
import os

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric

from groq_judge import GroqJudge

_judge = None

METRICS = [
    ("Faithfulness", FaithfulnessMetric, 0.7),
    ("Answer Relevancy", AnswerRelevancyMetric, 0.7),
    ("Contextual Relevancy", ContextualRelevancyMetric, 0.6),
]


def _get_judge():
    global _judge
    if _judge is None:
        _judge = GroqJudge()
    return _judge


def evaluate_answer(question: str, answer: str, retrieval_context: list[str]) -> list[dict]:
    """Score a live RAG answer. Returns [] if GROQ_API_KEY is not configured."""
    if not os.environ.get("GROQ_API_KEY") or not retrieval_context:
        return []

    test_case = LLMTestCase(input=question, actual_output=answer, retrieval_context=retrieval_context)
    judge = _get_judge()

    results = []
    for name, metric_cls, threshold in METRICS:
        try:
            metric = metric_cls(threshold=threshold, model=judge, include_reason=False)
            metric.measure(test_case)
            results.append({"metric": name, "score": float(metric.score), "success": bool(metric.success)})
        except Exception:
            continue
    return results
