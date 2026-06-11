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
from citations import dedupe_sources
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


class SourceItem(BaseModel):
    section: str
    page: str
    snippet: str = ""           # excerpt of the retrieved chunk, for citation previews
    file: str = ""              # source PDF filename, served via /documents


class QueryResponse(BaseModel):
    answer: str
    run_id: str                # use this to submit feedback
    sources: list[SourceItem]  # retrieved sections for transparency


REASON_CHOICES = {"incorrect", "incomplete", "not_relevant", "unclear", "other"}


class FeedbackRequest(BaseModel):
    run_id: str
    score: int                 # 1 = thumbs up, 0 = thumbs down
    reason: str | None = None  # required when score == 0, one of REASON_CHOICES
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
    sources: list[SourceItem]
    run_id: str | None
    created_at: str
    version_count: int = 1
    active_version: int = 1


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut]


class RenameConversationRequest(BaseModel):
    title: str


class VersionItem(BaseModel):
    version_index: int
    content: str
    sources: list[SourceItem]
    run_id: str | None
    created_at: str


class VersionsResponse(BaseModel):
    message_id: str
    versions: list[VersionItem]
    active_version: int


class ActivateVersionResponse(BaseModel):
    status: str
    active_version: int


def _make_title(question: str) -> str:
    """Derive a short conversation title from the first user question."""
    title = " ".join(question.strip().split())
    return title[:50] + ("…" if len(title) > 50 else "")


# ── In-memory run store (maps run_id → question + answer for LangSmith log) ──
_runs: dict[str, dict] = {}


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Run the RAG pipeline and return the answer with a run_id for feedback."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    chunks = retrieve_and_rerank(request.question)
    sources, citation_numbers = dedupe_sources(chunks)
    answer = generate(request.question, chunks, citation_numbers)

    run_id = str(uuid.uuid4())

    # Store run for feedback reference
    _runs[run_id] = {
        "question": request.question,
        "answer": answer,
        "sources": sources,
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

    return QueryResponse(answer=answer, run_id=run_id, sources=sources)


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    """Log thumbs up/down + optional text feedback to LangSmith."""
    if request.score not in (0, 1):
        raise HTTPException(status_code=400, detail="score must be 0 (down) or 1 (up)")

    if request.score == 0 and request.reason not in REASON_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"reason is required for negative feedback and must be one of {sorted(REASON_CHOICES)}",
        )

    if request.run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"run_id '{request.run_id}' not found")

    feedback_id = str(uuid.uuid4())

    run = _runs[request.run_id]
    now = datetime.utcnow().isoformat()

    db.add_feedback(
        feedback_id, request.run_id, request.score, request.reason, request.comment,
        run["question"], run["answer"], run["sources"], now,
    )

    # Also send to LangSmith if configured
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        try:
            ls = LangSmithClient()
            comment = request.comment or None
            if request.score == 0 and request.reason:
                comment = f"[{request.reason}] {comment or ''}".strip()
            ls.create_feedback(
                run_id=request.run_id,
                key="user_feedback",
                score=float(request.score),
                comment=comment,
                feedback_id=feedback_id,
            )
        except Exception:
            pass  # local save already succeeded

    return FeedbackResponse(status="ok", feedback_id=feedback_id)


@app.get("/api/feedback")
def get_feedback():
    """Retrieve all saved feedback entries from the database."""
    entries = db.list_feedback()
    return {"feedback": entries, "total": len(entries)}


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
    citation_numbers = []
    if classification == "rag":
        chunks = retrieve_and_rerank(request.question)
        sources, citation_numbers = dedupe_sources(chunks)

    _runs[run_id] = {"question": request.question, "answer": "", "sources": sources}
    assistant_message_id = str(uuid.uuid4())

    def event_generator():
        yield f"data: {json.dumps({'type': 'meta', 'run_id': run_id, 'conversation_id': conversation_id, 'message_id': assistant_message_id, 'sources': sources})}\n\n"
        full = []
        if classification == "rag":
            token_gen = generate_stream(request.question, chunks, citation_numbers)
        elif classification == "greeting":
            token_gen = generate_chat_stream(request.question)
        else:
            token_gen = generate_off_topic_stream()
        for token in token_gen:
            full.append(token)
            yield f"data: {json.dumps({'type': 'chunk', 'content': token})}\n\n"
        answer = "".join(full)
        _runs[run_id]["answer"] = answer
        db.add_message(assistant_message_id, conversation_id, "assistant", answer, sources, run_id, datetime.utcnow().isoformat())
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/messages/{message_id}/regenerate")
def regenerate_message(message_id: str):
    """Re-run the pipeline for an assistant message's preceding question and
    stream a new version, appending it to that message's version history.

    Event sequence mirrors /api/query/stream, with an extra `version_index`
    in the `meta` event and `total_versions` in the `done` event.
    """
    message = db.get_message(message_id)
    if message is None or message["role"] != "assistant":
        raise HTTPException(status_code=404, detail=f"assistant message '{message_id}' not found")

    user_message = db.get_preceding_user_message(message["conversation_id"], message_id)
    if user_message is None:
        raise HTTPException(status_code=404, detail="no preceding user message found")

    question = user_message["content"]

    # Lazily seed version 1 from the message's current content if it predates
    # version tracking.
    existing_versions = db.get_message_versions(message_id)
    if not existing_versions:
        db.add_message_version(
            str(uuid.uuid4()), message_id, 1,
            message["content"], message["sources"], message["run_id"], message["created_at"],
        )
        existing_versions = db.get_message_versions(message_id)

    next_index = len(existing_versions) + 1

    run_id = str(uuid.uuid4())
    classification = classify_query(question)

    sources = []
    chunks = []
    citation_numbers = []
    if classification == "rag":
        chunks = retrieve_and_rerank(question)
        sources, citation_numbers = dedupe_sources(chunks)

    _runs[run_id] = {"question": question, "answer": "", "sources": sources}

    def event_generator():
        yield f"data: {json.dumps({'type': 'meta', 'run_id': run_id, 'message_id': message_id, 'version_index': next_index, 'sources': sources})}\n\n"
        full = []
        if classification == "rag":
            token_gen = generate_stream(question, chunks, citation_numbers)
        elif classification == "greeting":
            token_gen = generate_chat_stream(question)
        else:
            token_gen = generate_off_topic_stream()
        for token in token_gen:
            full.append(token)
            yield f"data: {json.dumps({'type': 'chunk', 'content': token})}\n\n"
        answer = "".join(full)
        now = datetime.utcnow().isoformat()
        _runs[run_id]["answer"] = answer
        db.add_message_version(str(uuid.uuid4()), message_id, next_index, answer, sources, run_id, now)
        db.update_message(message_id, answer, sources, run_id, now, next_index)
        yield f"data: {json.dumps({'type': 'done', 'total_versions': next_index})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/messages/{message_id}/versions", response_model=VersionsResponse)
def get_message_versions(message_id: str):
    """List all stored versions of an assistant message, plus which is active."""
    message = db.get_message(message_id)
    if message is None:
        raise HTTPException(status_code=404, detail=f"message '{message_id}' not found")

    versions = db.get_message_versions(message_id)
    if not versions:
        versions = [{
            "version_index": 1,
            "content": message["content"],
            "sources": message["sources"],
            "run_id": message["run_id"],
            "created_at": message["created_at"],
        }]

    return VersionsResponse(message_id=message_id, versions=versions, active_version=message["active_version"])


@app.post("/api/messages/{message_id}/versions/{version_index}/activate", response_model=ActivateVersionResponse)
def activate_message_version(message_id: str, version_index: int):
    """Make a previously generated version the message's active content."""
    message = db.get_message(message_id)
    if message is None:
        raise HTTPException(status_code=404, detail=f"message '{message_id}' not found")

    versions = db.get_message_versions(message_id)
    if not versions:
        if version_index != 1:
            raise HTTPException(status_code=404, detail=f"version {version_index} not found")
        return ActivateVersionResponse(status="ok", active_version=1)

    match = next((v for v in versions if v["version_index"] == version_index), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"version {version_index} not found")

    db.update_message(message_id, match["content"], match["sources"], match["run_id"], match["created_at"], version_index)
    return ActivateVersionResponse(status="ok", active_version=version_index)


@app.get("/api/health")
def health():
    langsmith_enabled = os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    return {
        "status": "ok",
        "langsmith": langsmith_enabled,
        "project": os.environ.get("LANGCHAIN_PROJECT", "Dar_RAG") if langsmith_enabled else None,
    }
