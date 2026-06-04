# Local RAG Pipeline — Technical Report
**Project:** Dar@Week1 · Internal Ship Program  
**Document:** CIS Critical Security Controls v8 (82 pages, scanned PDF)  
**Stack:** Python 3.13 · LangChain · Weaviate · Ollama · DeepEval  
**Repository:** https://github.com/assi09/Dar_RAG

---

## 1. Project Overview

This project implements a fully local Retrieval-Augmented Generation (RAG) pipeline over the CIS Controls v8 document. The goal is to answer natural language security questions grounded strictly in the document — with no API keys, no cloud services, and no data leaving the machine. Every component from OCR parsing to answer generation runs on CPU.

RAG addresses a fundamental LLM limitation: models cannot know private or domain-specific documents. Instead of fine-tuning, RAG retrieves the relevant document fragments at query time and passes them as context to the generation model — giving a small local model the right information to answer accurately.

The pipeline follows two phases:
- **Offline (ingestion):** Parse → Chunk → Embed → Store
- **Online (query):** Embed query → Retrieve → Rerank → Generate

---

## 2. Project Framework & Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.13 | All pipeline code |
| Orchestration | LangChain | 1.3.2 | Document loaders, splitters, retrievers, chains |
| PDF Parsing | unstructured[pdf] | 0.22+ | OCR + layout detection on scanned PDFs |
| OCR Engine | Tesseract | system | Character recognition from page images |
| Layout Model | yolox_l0.05.onnx | 217MB | Page region detection (title, paragraph, table) |
| Embedding Model | BAAI/bge-small-en-v1.5 | 133MB | 384-dim vectors, CPU, normalized |
| Reranking Model | BAAI/bge-reranker-v2-m3 | 2.27GB | Cross-encoder relevance scoring |
| Vector Database | Weaviate | 1.37.7 | Hybrid search (vector + BM25), Docker |
| LLM (Generation) | llama3.2:3B | 2.0GB | Answer generation via Ollama |
| LLM (Inference Server) | Ollama | 0.12.10 | Serves llama3.2 locally on port 11434 |
| Evaluation | DeepEval | 4.0.5 | 6-metric RAG evaluation framework |
| Eval Judge (cloud) | Groq llama-3.3-70b | API | Reliable judge for DeepEval metrics |
| Sync | LangChain Indexing API | — | Incremental re-ingestion with SQLite tracking |
| Environment | Python venv | — | Isolated dependency environment |
| Version Control | Git + GitHub | — | Repository: assi09/Dar_RAG |

---

## 3. Pipeline Stages

### Stage 2.1 — Parsing

**What it does:** Converts the raw PDF into structured text elements with metadata.

The CIS Controls PDF is a scanned document — it has no embedded text layer, making standard extraction impossible. The pipeline uses `UnstructuredLoader` from `langchain-unstructured` with `strategy="hi_res"` and `mode="elements"`. This triggers a three-step process: (1) YOLO layout detection identifies regions on each page image (title, paragraph, table, list); (2) Tesseract OCR reads text from those regions; (3) each element gets a category tag and page-level metadata.

The extracted elements are then grouped into sections via a custom `group_by_section()` function. It walks elements in document order and starts a new section whenever it encounters a `Title` element. A critical decision was introduced: numbered safeguard titles (e.g., `13.1 Centralize Security Event Alerting`) are **not** treated as section boundaries — they are instead appended as content to their parent control section. Without this, each safeguard became its own micro-section, producing 325 isolated chunks averaging 103 characters — far too small for quality embeddings. After the fix, sections dropped to 131, and chunks became semantically richer.

Front matter sections (Contents, Glossary, Acronyms) are filtered out entirely since they contain no answer-worthy content. OCR noise from colored table cells — artifacts like `@EEEED`, `lelele`, `t Protect ]` from the safeguard table's IG1/IG2/IG3 dot columns — is cleaned with targeted regex before being stored.

**Output:** 131 meaningful sections, each covering one control or sub-topic.

---

### Stage 2.2 — Chunking

**What it does:** Splits sections into smaller overlapping passages for embedding.

`RecursiveCharacterTextSplitter` splits each section with a maximum of **1,400 characters** (~350 tokens) and **10% overlap** (140 chars). The splitter tries cut points in priority order: `\n\n` → `\n` → `.` → space, always choosing the most natural boundary.

**Why 1,400 characters:** The embedding model `bge-small-en-v1.5` has a 512-token context window. At roughly 4 chars per token, 512 tokens ≈ 2,048 chars. The recommended sweet spot for RAG is 256–400 tokens per chunk; 1,400 chars ≈ 350 tokens, placing us solidly in that range. An initial value of 512 chars was used but this was only ~128 tokens — underutilising the model's capacity.

Every chunk has the section title prepended in brackets — e.g., `[Control 12: Network Infrastructure Management]` — so that when retrieved in isolation, the reranker and generator see the section context directly. Chunks shorter than 50 characters are dropped as noise.

**Output:** 234 clean chunks, average ~530 chars, stored with `section_title`, `page_number`, and `filename` metadata.

---

### Stage 2.3 — Embedding

**What it does:** Converts each chunk of text into a 384-dimensional numeric vector.

`BAAI/bge-small-en-v1.5` is used via `langchain-huggingface`. The model runs entirely on CPU, downloads once (~133MB) and caches permanently at `~/.cache/huggingface/hub/`. The key parameter is `normalize_embeddings=True` — required for bge models. Without normalization, cosine similarity scores are numerically unreliable and retrieval ranking breaks.

The embedding model converts text into a point in 384-dimensional space. Semantically similar text produces vectors that are geometrically close — this is what makes semantic search possible, allowing queries to match relevant chunks even when they use different vocabulary than the document.

---

### Stage 2.4 — Ingesting (Vector Store)

**What it does:** Stores the embedded chunks in Weaviate for fast retrieval.

Weaviate runs in Docker on port 8080 and persists data in volumes across restarts. The `langchain-weaviate` wrapper provides a LangChain-compatible interface. The collection is named `DarRAG`.

Weaviate stores both the 384-dim vector and the original chunk text, enabling **hybrid search** — a blend of cosine vector similarity and BM25 keyword matching. The blend ratio (`alpha`) was empirically tested across values 0.0 to 1.0 against the golden dataset; all alphas achieved 10/10 section hit rate, confirming the reranker compensates for search quality differences at this scale.

A critical operational bug was fixed: the `ingest()` function now **deletes the existing collection before inserting**, preventing silent accumulation across re-runs. Without this, multiple ingest runs stacked to 1,611 chunks (4× expected), causing deduplication failures and degraded retrieval.

**Output:** 234 chunks stored in Weaviate, ingestion time ~5 seconds (after OCR completes).

---

### Stage 2.5 — Sync

**What it does:** Enables incremental re-ingestion when source documents change.

A two-layer approach was implemented. The outer layer computes an MD5 hash of each PDF file's raw bytes and compares against stored hashes in `data/.file_hashes.json`. If the file is unchanged, the entire pipeline is skipped — no OCR, no embedding, instant return. This is necessary because hi-res OCR is non-deterministic: the same PDF produces slightly different text across runs, making content-hash-based deduplication unreliable.

The inner layer uses the **LangChain Indexing API** with `SQLRecordManager` (SQLite) to track chunk hashes and handle incremental updates and deletions for files that did change.

---

### Stage 2.7 — Retrieval

**What it does:** Embeds the user query and finds the most similar chunks.

The query is embedded using the same `bge-small-en-v1.5` model used during ingestion — this is essential: both query and document vectors must live in the same vector space for distance comparisons to be meaningful. Weaviate returns the top **50 candidates** (`k=50` as specified in the reference materials).

Before reranking, a deduplication step removes near-duplicate chunks using a fingerprint of section title + first 80 characters of content. This handles OCR-variant duplicates where the appendix (which repeats all safeguard tables) produces chunks with slightly different text from the main content, making exact-hash deduplication ineffective. Without this, appendix duplicates occupied 3 of 5 reranked slots, pushing relevant main-content chunks out.

---

### Stage 2.6 — Reranking

**What it does:** Re-scores the 50 candidates using a more powerful cross-encoder model.

`BAAI/bge-reranker-v2-m3` (568M parameters) reads each (query, chunk) pair **together** and outputs a relevance score. Unlike the bi-encoder used in retrieval — which encodes query and document independently — the cross-encoder sees both simultaneously and captures fine-grained interaction. This distinction matters: a query like "network security controls" is semantically closer to generic security overview text than to specific safeguard entries in pure vector space, but the cross-encoder correctly identifies the specific safeguards as more relevant.

The reranker returns the **top 5 chunks** after scoring. These 5 chunks are what the generator receives as context.

**Timing:** embedding + retrieval is sub-second; reranking adds ~500ms on CPU.

---

### Stage 2.7 — Generation

**What it does:** Generates a natural language answer from the retrieved context.

`llama3.2:3B` runs via Ollama (localhost:11434). `temperature=0.0` ensures deterministic, non-creative answers. The system prompt enforces strict context-only answering with explicit instructions to synthesize across all retrieved sections and only cite safeguard numbers explicitly present in the context.

The prompt assembles all 5 chunks separated by `---` markers. Since every chunk already starts with `[Section Title]`, the generator has structural context for each passage. An earlier bug was corrected where `build_prompt()` was adding a redundant `[Section: X | Page: Y]` header on top of the existing title prefix, causing each safeguard entry to appear twice in the context window.

**Throughput:** ~3–6 tokens/second on CPU (MacBook Pro M-series). Adequate for single-query use; not suitable for real-time chat.

---

## 4. Evaluation Framework

### 4.1 Golden Dataset

A 10-entry golden dataset was manually constructed from the CIS Controls PDF (`data/golden_dataset.json`). Each entry contains:
- `question` — the test query
- `ground_truth_answer` — the correct answer extracted directly from the PDF
- `ground_truth_context` — the source passage
- `source_pages` — expected PDF page numbers
- `keywords` — specific terms that must appear in retrieved chunks
- `control_number` — the CIS control being tested

This dataset serves as a reproducible benchmark for evaluating both retrieval quality (keyword score, section hit rate) and generation quality (via DeepEval metrics).

### 4.2 Custom Evaluator

`evaluate.py` runs all 10 questions through the retrieval pipeline and measures two metrics without LLM involvement:

- **Section Hit Rate:** did we retrieve at least one chunk from the expected pages?
- **Keyword Score:** what fraction of expected keywords appeared in retrieved chunks?

**Results after all pipeline improvements:**
```
Section Hit Rate : 10/10 (100%)
Avg Keyword Score: 82%
```

### 4.3 DeepEval — Local Judge (llama3.2)

`eval_deepeval.py` runs 5 sampled questions through the full pipeline (retrieve → rerank → generate) and scores the outputs using 6 metrics with llama3.2 as judge:

| Metric | Avg Score | Pass Rate |
|---|---|---|
| Faithfulness | 0.67 | 40% |
| Answer Relevancy | 0.97 | 100% |
| Contextual Precision | 0.92 | 100% |
| Contextual Recall | 0.79 | 80% |

The low Faithfulness pass rate (40%) is not a pipeline failure — it reflects the judge's (llama3.2 3B) inconsistency. In multiple cases the judge stated "no contradictions found" while assigning a failing score of 0.50–0.60, directly contradicting its own reasoning. A 3B model is too small to reliably evaluate complex semantic relationships.

### 4.4 DeepEval — Groq Judge (llama-3.3-70b)

`eval_deepeval_groq.py` uses the same pipeline but replaces the judge with `llama-3.3-70b-versatile` via the Groq API:

```
✅ test_case_0 (Passed 4 metrics)
✅ test_case_1 (Passed 4 metrics)
✅ test_case_2 (Passed 4 metrics)
✅ test_case_3 (Passed 4 metrics)
✅ test_case_4 (Passed 4 metrics)

Faithfulness         0.97  100%
Answer Relevancy     1.00  100%
Contextual Precision 0.99  100%
Contextual Recall    1.00  100%

Pass Rate: 100% | Time: 220.82s
```

The difference between llama3.2 judge (40% faithfulness pass) and Groq 70B judge (100% pass) proves the earlier failures were judge quality issues, not pipeline defects.

### 4.5 Synthesizer (Auto-Generated Dataset)

`eval_deepeval_synth.py` uses DeepEval's `Synthesizer` to auto-generate Q&A pairs from Weaviate chunks using Groq, wrapping them in DeepEval's `EvaluationDataset` format. This reduces the manual effort of golden dataset curation and enables A/B testing between pipeline configurations.

### 4.6 Pytest Integration

`test_rag.py` implements pytest-compatible tests using `@pytest.mark.parametrize` and `assert_test`. Tests can be run with `deepeval test run src/test_rag.py` to integrate evaluation into CI/CD pipelines.

---

## 5. Key Results

### End-to-End Test (8 representative questions)

| Question | Verdict |
|---|---|
| How often should audit logs be retained? | ✅ Correct — "minimum 90 days" |
| Minimum password length without MFA? | ✅ Correct — "14 characters" |
| Penetration testing frequency? | ✅ Correct — "no less than annually" |
| How should vulnerability management be implemented? | ✅ Correct — lists 7.1–7.5 with SCAP detail |
| What safeguards protect against malware? | ✅ Correct — lists 9.4, 9.6, 9.7, 10.5–10.7 |
| Control 12 safeguards for network infrastructure? | ✅ Correct — all 8 safeguards (12.1–12.8) |
| How to monitor network traffic for threats? | ⚠️ Partial — cites 13.6–13.8, misses 13.1 (SIEM) |
| How should enterprise assets be inventoried? | ✅ Correct — lists 1.1 details + 1.2 |

**7/8 correct.** The one partial answer (network monitoring) retrieves the right control but favours the narrative section over the safeguards table containing 13.1 (SIEM) — the most critical safeguard.

---

## 6. Limitations

**1. Generation completeness vs. retrieval correctness.**
Retrieval lands on the correct control for all tested questions (10/10 section hit rate). However, when a control has many safeguards (Control 13 has 11), the generator receives only 5 chunks — not all safeguard entries are present. The generator produces partially complete answers even when the retrieved content is correct.

**2. Small LLM as generation model.**
llama3.2 at 3B parameters is CPU-feasible but limited. It does not reliably synthesize across all 5 retrieved chunks, tends to stop early, and occasionally misattributes safeguard numbers. A 7B+ model would produce materially better generation quality.

**3. OCR non-determinism.**
The CIS Controls PDF is fully scanned (no embedded text). Each hi-res OCR run produces slightly different output — the same sentence may read differently across two runs. This makes content-hash deduplication imperfect and means exact reproducibility of chunk text is not guaranteed.

**4. Parent-child chunking not implemented.**
The current architecture uses flat chunks. For "enumeration" questions requiring all safeguards of a control, the answer is structurally incomplete. Parent-child retrieval — where small child chunks are embedded for precision and the full parent section is returned to the generator — would address this as a Week 2 improvement.

**5. Groq daily token limit.**
The free tier for `llama-3.3-70b-versatile` is capped at 100,000 tokens per day. The synthesizer and evaluation suite together can exhaust this in a single session. Fallback model `llama-3.1-8b-instant` (500K TPD) is configured in `.env`.

---

## 7. File Structure

```
Dar@Week1/
├── src/
│   ├── parser.py          # OCR parsing, section grouping, OCR cleaning
│   ├── chunker.py         # RecursiveCharacterTextSplitter + title prefix
│   ├── embedder.py        # bge-small-en-v1.5 wrapper
│   ├── vector_store.py    # Weaviate connection + ingest
│   ├── ingest.py          # Full ingestion pipeline (parse→chunk→embed→store)
│   ├── sync.py            # Incremental sync (file hash + Indexing API)
│   ├── retriever.py       # Basic vector retrieval
│   ├── reranker.py        # Two-stage retrieval + cross-encoder reranking
│   ├── generator.py       # llama3.2 generation via Ollama
│   ├── main.py            # End-to-end entrypoint (single query + interactive)
│   ├── evaluate.py        # Custom golden dataset evaluator (no LLM judge)
│   ├── groq_judge.py      # Shared GroqJudge model for DeepEval
│   ├── eval_deepeval.py   # DeepEval evaluation (Ollama judge)
│   ├── eval_deepeval_groq.py  # DeepEval evaluation (Groq judge)
│   ├── eval_deepeval_synth.py # DeepEval + Synthesizer (auto-generated dataset)
│   ├── generate_goldens.py    # Standalone golden dataset generator
│   └── test_rag.py            # pytest integration
├── data/
│   ├── docs/              # Source PDFs (gitignored)
│   └── golden_dataset.json    # 10 hand-crafted Q&A pairs
├── rag_setup/             # Reference materials (PPTX, notebook, pyproject.toml)
├── requirements.txt
├── .env                   # API keys (gitignored)
└── .gitignore
```

---

## 8. How to Run

```bash
# 1. Start Weaviate
docker run -d --name weaviate -p 8080:8080 -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest

# 2. Activate environment
source venv/bin/activate

# 3. Ingest the document (one-time, ~4 mins OCR)
HF_HUB_DISABLE_XET=1 python3 src/ingest.py

# 4. Ask a question
HF_HUB_DISABLE_XET=1 python3 src/main.py "What are the safeguards for audit log management?"

# 5. Run retrieval evaluation (no API key needed)
HF_HUB_DISABLE_XET=1 python3 src/evaluate.py

# 6. Run DeepEval evaluation (Groq API key in .env)
HF_HUB_DISABLE_XET=1 python3 src/eval_deepeval_groq.py
```
