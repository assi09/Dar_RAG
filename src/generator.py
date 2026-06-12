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
- Be concise. No filler phrases like "Based on the context..." or "According to the document...".
- Each context block below is tagged with a citation number like [1], [2], etc. When you use information from a block, cite it inline immediately after the relevant sentence or clause using its number, e.g. "Audit logs must be retained for 90 days [1]."
- Always reuse the SAME number for the same context block. Never invent a citation number that isn't shown below.
- If multiple blocks support a statement, cite all of them, e.g. [1][3]."""


CHAT_SYSTEM_PROMPT = """You are Dar, a friendly assistant for a CIS Controls v8 knowledge base.

The user just sent a casual message (greeting, thanks, small talk, or a question about
what you can do) rather than a question about CIS Controls v8. Reply briefly and warmly
in plain conversational language — do not mention "context" or "documents". If it fits
naturally, mention that you can answer questions about CIS Controls v8 safeguards."""


CLASSIFIER_PROMPT = """Classify the user's message into exactly one of three categories:

- "rag": a question about CIS Controls v8, cybersecurity safeguards, or related \
security/compliance topics that should be answered from a reference document.
- "greeting": casual greetings, thanks, goodbyes, or small talk directed at the \
assistant itself (e.g. "hi", "thanks", "how are you", "what can you do").
- "off_topic": anything else — general knowledge questions or requests unrelated to \
CIS Controls v8 or security (e.g. math, science, history, coding help, other subjects).

Respond with exactly one word: "rag", "greeting", or "off_topic", and nothing else.

Message: {message}"""


OFF_TOPIC_REPLY = (
    "I'm a CIS Controls v8 assistant, so I can only help with questions about "
    "those security controls and safeguards. Try asking something like "
    "\"What does Control 8 say about audit log retention?\""
)


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)


def build_prompt(query: str, chunks: list[dict], citation_numbers: list[int]) -> list:
    # chunk content already starts with [section_title] from chunker.py
    # so we just join them — no need to add another header wrapper.
    # Group chunks by citation number so the LLM sees one block per [N] tag,
    # even if multiple retrieved chunks map to the same deduplicated source.
    grouped: dict[int, list[str]] = {}
    for chunk, num in zip(chunks, citation_numbers):
        grouped.setdefault(num, []).append(chunk['content'])

    blocks = [f"[{num}] " + "\n\n".join(parts) for num, parts in grouped.items()]
    context = "\n\n---\n\n".join(blocks)

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]


def generate(query: str, chunks: list[dict], citation_numbers: list[int]) -> str:
    """Generate an answer from retrieved chunks using llama3.2."""
    llm = get_llm()
    messages = build_prompt(query, chunks, citation_numbers)
    response = llm.invoke(messages)
    return response.content


def generate_stream(query: str, chunks: list[dict], citation_numbers: list[int], run_id=None):
    """Yield tokens from llama3.2 one at a time for SSE streaming.

    Passing run_id sets the LangSmith trace ID for this LLM call, so that
    feedback submitted with the same ID attaches to this trace.
    """
    llm = get_llm()
    messages = build_prompt(query, chunks, citation_numbers)
    config = {"run_id": run_id} if run_id else {}
    for token in llm.stream(messages, config=config):
        if token.content:
            yield token.content


def classify_query(query: str) -> str:
    """Classify a message as 'rag', 'greeting', or 'off_topic'."""
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=CLASSIFIER_PROMPT.format(message=query))])
    label = response.content.strip().lower()
    if "off_topic" in label or "off-topic" in label:
        return "off_topic"
    if "greeting" in label:
        return "greeting"
    return "rag"


def generate_chat_stream(query: str, run_id=None):
    """Yield tokens for a casual conversational reply (no document context)."""
    llm = get_llm()
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT), HumanMessage(content=query)]
    config = {"run_id": run_id} if run_id else {}
    for token in llm.stream(messages, config=config):
        if token.content:
            yield token.content


def generate_off_topic_stream():
    """Yield a fixed redirect reply for non-CIS-Controls questions, word by word."""
    words = OFF_TOPIC_REPLY.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")


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
    answer = generate(query, test_chunks, citation_numbers=[1])
    print(f"Answer:\n{answer}")
