from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

LLM_MODEL = "llama3.2"
TEMPERATURE = 0.0  # deterministic — RAG answers should not be creative


SYSTEM_PROMPT = """You are a precise assistant that answers questions strictly based on the provided context.

Rules:
- Answer ONLY from the context below. Do not use outside knowledge.
- Read ALL provided sections carefully before answering — the answer may be spread across multiple sections.
- Synthesize relevant information from every section that addresses the question. Do not stop after the first section.
- Ignore sections that are not relevant to the question — do not include unrelated content just because it was retrieved.
- Only say "I don't have enough information" if the context contains absolutely nothing related to the question.
- When listing safeguards or steps, use a numbered list. Only include safeguard numbers (e.g. 12.1, 13.3) if they are explicitly stated in the context — never invent them.
- Be concise. No filler phrases like "Based on the context..." or "According to the document..."."""


CHAT_SYSTEM_PROMPT = """You are Dar, a friendly assistant for a CIS Controls v8 knowledge base.

The user just sent a casual message (greeting, thanks, small talk, or a question about
what you can do) rather than a question about CIS Controls v8. Reply briefly and warmly
in plain conversational language — do not mention "context" or "documents". If it fits
naturally, mention that you can answer questions about CIS Controls v8 safeguards."""


CLASSIFIER_PROMPT = """Classify the user's message as either "rag" or "chat".

- "rag": the message asks a question about CIS Controls v8, cybersecurity safeguards, \
or related security/compliance topics that should be answered from a reference document.
- "chat": the message is casual conversation — greetings, thanks, small talk, or a \
general question about the assistant itself (not requiring document lookup).

Respond with exactly one word, "rag" or "chat", and nothing else.

Message: {message}"""


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)


def build_prompt(query: str, chunks: list[dict]) -> list:
    # chunk content already starts with [section_title] from chunker.py
    # so we just join them — no need to add another header wrapper
    context = "\n\n---\n\n".join(c['content'] for c in chunks)

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]


def generate(query: str, chunks: list[dict]) -> str:
    """Generate an answer from retrieved chunks using llama3.2."""
    llm = get_llm()
    messages = build_prompt(query, chunks)
    response = llm.invoke(messages)
    return response.content


def generate_stream(query: str, chunks: list[dict]):
    """Yield tokens from llama3.2 one at a time for SSE streaming."""
    llm = get_llm()
    messages = build_prompt(query, chunks)
    for token in llm.stream(messages):
        if token.content:
            yield token.content


def classify_query(query: str) -> str:
    """Classify a message as 'rag' (needs document retrieval) or 'chat' (casual)."""
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=CLASSIFIER_PROMPT.format(message=query))])
    label = response.content.strip().lower()
    return "chat" if "chat" in label else "rag"


def generate_chat_stream(query: str):
    """Yield tokens for a casual conversational reply (no document context)."""
    llm = get_llm()
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT), HumanMessage(content=query)]
    for token in llm.stream(messages):
        if token.content:
            yield token.content


if __name__ == "__main__":
    # Quick test with hardcoded context
    test_chunks = [
        {
            "section": "Control 08: Audit Log Management",
            "page": 37,
            "content": (
                "8.1 Establish and Maintain an Audit Log Management Process — "
                "Establish and maintain an audit log management process that defines the enterprise's "
                "logging requirements. At a minimum, address the collection, review, and retention of "
                "audit logs for enterprise assets.\n"
                "8.10 Retain Audit Logs — Retain audit logs across enterprise assets for a minimum of 90 days.\n"
                "8.11 Conduct Audit Log Reviews — Conduct reviews of audit logs to detect anomalies or "
                "abnormal events that could indicate a potential threat. Conduct reviews on a weekly, "
                "or more frequent, basis."
            ),
        }
    ]

    query = "How long should audit logs be retained?"
    print(f"Query: {query}\n")
    answer = generate(query, test_chunks)
    print(f"Answer:\n{answer}")
