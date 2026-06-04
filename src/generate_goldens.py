"""
Golden Dataset Generation using DeepEval Synthesizer + Groq API.
Auto-generates question-answer pairs from our parsed document chunks.

Usage:
  export GROQ_API_KEY=gsk_...
  python3 src/generate_goldens.py
"""
import json
import os
import sys
sys.path.insert(0, 'src')

from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

from groq_judge import GroqJudge
from embedder import get_embedder
from vector_store import get_client, get_vector_store

OUTPUT_DIR = "data"
MAX_GOLDENS = 20


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY before running.")
        sys.exit(1)

    print("Loading chunks from Weaviate (no re-parsing needed)...")
    embedder = get_embedder()
    client   = get_client()
    store    = get_vector_store(client, embedder)

    # Fetch a broad sample of chunks from Weaviate using a generic query
    results = store.similarity_search("CIS Controls safeguards security enterprise", k=MAX_GOLDENS)
    client.close()

    chunks = [r for r in results if len(r.page_content.strip()) > 100]
    print(f"Loaded {len(chunks)} chunks as source material\n")

    # Each context is a list of strings — one chunk per golden
    contexts = [[chunk.page_content] for chunk in chunks]
    print(f"Using {len(contexts)} contexts for generation\n")

    judge = GroqJudge()
    synthesizer = Synthesizer(model=judge)

    print(f"Generating up to {MAX_GOLDENS} golden Q&A pairs using {judge.get_model_name()}...")
    synthesizer.generate_goldens(
        contexts=contexts[:MAX_GOLDENS],  # generate one golden per context
        include_expected_output=True,
    )

    dataset = EvaluationDataset(goldens=synthesizer.synthetic_goldens)
    dataset.save_as(file_type="json", directory=OUTPUT_DIR)
    print(f"\nSaved {len(dataset.goldens)} goldens to {OUTPUT_DIR}/")

    # Also preview a few
    print("\nSample generated goldens:")
    for g in dataset.goldens[:3]:
        print(f"\n  Q: {g.input}")
        print(f"  A: {g.expected_output[:150] if g.expected_output else 'N/A'}")


if __name__ == "__main__":
    main()
