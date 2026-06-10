"""
3.3 HITL Feedback API — FastAPI + LangSmith

Endpoints:
  POST /api/query    — run the RAG pipeline, returns answer + run_id
  POST /api/feedback — log thumbs up/down + text feedback to LangSmith

Setup:
  Add to .env:
    LANGSMITH_API_KEY=ls__...   (get from https://smith.langchain.com)
    LANGCHAIN_PROJECT=Dar_RAG

  Run:
    HF_HUB_DISABLE_XET=1 uvicorn src.api:app --reload --port 8000
"""
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")

# Enable LangSmith tracing if API key is present
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "Dar_RAG")

sys.path.insert(0, str(Path(__file__).parent))

from reranker import retrieve_and_rerank
from generator import generate, generate_stream, classify_query, generate_chat_stream, generate_off_topic_stream
from langsmith import Client as LangSmithClient
import db


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dar RAG API",
    description="CIS Controls v8 RAG pipeline with human feedback via LangSmith",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()


# ── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    run_id: str                # use this to submit feedback
    sources: list[dict]        # retrieved sections for transparency


class FeedbackRequest(BaseModel):
    run_id: str
    score: int                 # 1 = thumbs up, 0 = thumbs down
    comment: str = ""          # optional text feedback


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    sources: list[dict]
    run_id: str | None
    created_at: str


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut]


class RenameConversationRequest(BaseModel):
    title: str


def _make_title(question: str) -> str:
    """Derive a short conversation title from the first user question."""
    title = " ".join(question.strip().split())
    return title[:50] + ("…" if len(title) > 50 else "")


# ── Persistence ─────────────────────────────────────────────────────────────
FEEDBACK_FILE = Path(__file__).parent.parent / "data" / "feedback_log.json"


def _load_feedback() -> list[dict]:
    if FEEDBACK_FILE.exists():
        return json.loads(FEEDBACK_FILE.read_text())
    return []


def _save_feedback(entry: dict):
    log = _load_feedback()
    log.append(entry)
    FEEDBACK_FILE.write_text(json.dumps(log, indent=2))


# ── In-memory run store (maps run_id → question + answer for LangSmith log) ──
_runs: dict[str, dict] = {}


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Run the RAG pipeline and return the answer with a run_id for feedback."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    chunks = retrieve_and_rerank(request.question)
    answer = generate(request.question, chunks)

    run_id = str(uuid.uuid4())

    # Store run for feedback reference
    _runs[run_id] = {
        "question": request.question,
        "answer": answer,
        "sources": [{"section": c["section"], "page": c["page"]} for c in chunks],
    }

    # Log to LangSmith if tracing is enabled
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        try:
            ls = LangSmithClient()
            ls.create_run(
                id=run_id,
                name="rag_query",
                run_type="chain",
                inputs={"question": request.question},
                outputs={"answer": answer},
                project_name=os.environ.get("LANGCHAIN_PROJECT", "Dar_RAG"),
            )
        except Exception:
            pass  # feedback logging is non-blocking

    seen: set[tuple] = set()
    sources = []
    for c in chunks:
        key = (c["section"], int(float(c["page"])))
        if key not in seen:
            seen.add(key)
            sources.append({"section": c["section"], "page": str(key[1])})

    return QueryResponse(answer=answer, run_id=run_id, sources=sources)


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    """Log thumbs up/down + optional text feedback to LangSmith."""
    if request.score not in (0, 1):
        raise HTTPException(status_code=400, detail="score must be 0 (down) or 1 (up)")

    if request.run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"run_id '{request.run_id}' not found")

    feedback_id = str(uuid.uuid4())

    run = _runs[request.run_id]

    # Always save locally to data/feedback_log.json
    entry = {
        "feedback_id": feedback_id,
        "run_id": request.run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "score": request.score,
        "comment": request.comment,
        "question": run["question"],
        "answer": run["answer"],
        "sources": run["sources"],
    }
    _save_feedback(entry)

    # Also send to LangSmith if configured
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        try:
            ls = LangSmithClient()
            ls.create_feedback(
                run_id=request.run_id,
                key="user_feedback",
                score=float(request.score),
                comment=request.comment or None,
                feedback_id=feedback_id,
            )
        except Exception:
            pass  # local save already succeeded

    return FeedbackResponse(status="ok", feedback_id=feedback_id)


@app.get("/api/feedback")
def get_feedback():
    """Retrieve all saved feedback entries from data/feedback_log.json."""
    return {"feedback": _load_feedback(), "total": len(_load_feedback())}


@app.post("/api/conversations", response_model=ConversationSummary)
def create_conversation():
    """Start a new, empty conversation."""
    conversation_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    db.create_conversation(conversation_id, "New conversation", now)
    return ConversationSummary(id=conversation_id, title="New conversation", created_at=now, updated_at=now)


@app.get("/api/conversations", response_model=list[ConversationSummary])
def list_conversations():
    """List all conversations, most recently updated first."""
    return db.list_conversations()


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str):
    """Fetch a conversation and its full message history."""
    convo = db.get_conversation(conversation_id)
    if convo is None:
        raise HTTPException(status_code=404, detail=f"conversation '{conversation_id}' not found")
    return convo


@app.patch("/api/conversations/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(conversation_id: str, request: RenameConversationRequest):
    """Rename a conversation."""
    convo = db.get_conversation(conversation_id)
    if convo is None:
        raise HTTPException(status_code=404, detail=f"conversation '{conversation_id}' not found")

    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title must not be empty")

    now = datetime.utcnow().isoformat()
    db.rename_conversation(conversation_id, title, now)
    return ConversationSummary(id=conversation_id, title=title, created_at=convo["created_at"], updated_at=now)


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation and its messages."""
    if db.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail=f"conversation '{conversation_id}' not found")
    db.delete_conversation(conversation_id)
    return {"status": "ok"}


@app.post("/api/query/stream")
def query_stream(request: QueryRequest):
    """Stream the RAG answer as Server-Sent Events (text/event-stream).

    Event sequence:
      data: {"type": "meta",  "run_id": "...", "conversation_id": "...", "sources": [...]}
      data: {"type": "chunk", "content": "..."}   (repeated per token)
      data: {"type": "done"}
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    run_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    # Resume an existing conversation, or start a new one
    conversation_id = request.conversation_id
    if conversation_id is None or db.get_conversation(conversation_id) is None:
        conversation_id = str(uuid.uuid4())
        db.create_conversation(conversation_id, _make_title(request.question), now)

    db.add_message(str(uuid.uuid4()), conversation_id, "user", request.question, [], None, now)

    # Casual greetings skip retrieval and get a friendly reply; off-topic questions
    # get a fixed redirect; only "rag" messages hit the document pipeline — vector
    # search always returns "closest" chunks even when nothing is relevant.
    classification = classify_query(request.question)

    sources = []
    chunks = []
    if classification == "rag":
        chunks = retrieve_and_rerank(request.question)

        # Deduplicate by section+page (multiple chunks from the same section are
        # useful for generation but redundant in the sources display)
        seen_sources: set[tuple] = set()
        for c in chunks:
            key = (c["section"], int(float(c["page"])))
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({"section": c["section"], "page": str(key[1])})

    _runs[run_id] = {"question": request.question, "answer": "", "sources": sources}

    def event_generator():
        yield f"data: {json.dumps({'type': 'meta', 'run_id': run_id, 'conversation_id': conversation_id, 'sources': sources})}\n\n"
        full = []
        if classification == "rag":
            token_gen = generate_stream(request.question, chunks)
        elif classification == "greeting":
            token_gen = generate_chat_stream(request.question)
        else:
            token_gen = generate_off_topic_stream()
        for token in token_gen:
            full.append(token)
            yield f"data: {json.dumps({'type': 'chunk', 'content': token})}\n\n"
        answer = "".join(full)
        _runs[run_id]["answer"] = answer
        db.add_message(str(uuid.uuid4()), conversation_id, "assistant", answer, sources, run_id, datetime.utcnow().isoformat())
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
def health():
    langsmith_enabled = os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    return {
        "status": "ok",
        "langsmith": langsmith_enabled,
        "project": os.environ.get("LANGCHAIN_PROJECT", "Dar_RAG") if langsmith_enabled else None,
    }
