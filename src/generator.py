from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

LLM_MODEL = "llama3.2"
TEMPERATURE = 0.0  # deterministic — RAG answers should not be creative


SYSTEM_PROMPT = """You are a precise assistant that answers questions strictly based on the provided context.

Rules:
- Answer ONLY from the context below. Do not use outside knowledge.
- If ANY relevant information exists in the context, use it to answer — even if it is partial or spread across sections.
- Only say "I don't have enough information" if the context contains absolutely nothing related to the question.
- Be concise and direct. No filler phrases like "Based on the context..." or "According to the document...".
- When listing safeguards or steps, use a numbered list.
- Always cite the section name and safeguard number when referencing specific controls."""


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)


def build_prompt(query: str, chunks: list[dict]) -> list:
    context = "\n\n---\n\n".join(
        f"[Section: {c['section']} | Page: {c['page']}]\n{c['content']}"
        for c in chunks
    )

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
