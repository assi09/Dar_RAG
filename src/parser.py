from pathlib import Path
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document


DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
MIN_ELEMENT_LENGTH = 20  # filter OCR noise / stray characters


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Parse all documents and return section-grouped Documents ready for chunking."""
    files = list(docs_dir.glob("**/*"))
    files = [f for f in files if f.is_file() and f.suffix in {".pdf", ".txt", ".docx", ".md"}]

    if not files:
        raise FileNotFoundError(f"No supported documents found in {docs_dir}")

    print(f"Found {len(files)} file(s): {[f.name for f in files]}")

    all_elements: list[Document] = []
    for file_path in files:
        print(f"Parsing: {file_path.name} ...")
        loader = UnstructuredLoader(
            file_path=str(file_path),
            strategy="hi_res",
            mode="elements",
        )
        try:
            elements = loader.load()
        except Exception as e:
            print(f"  WARNING: parse error ({type(e).__name__}), retrying with fast strategy...")
            loader = UnstructuredLoader(
                file_path=str(file_path),
                strategy="fast",
                mode="elements",
            )
            elements = loader.load()
        print(f"  -> {len(elements)} element(s) extracted")
        all_elements.extend(elements)

    sections = group_by_section(all_elements)
    print(f"\nTotal sections after grouping: {len(sections)}")
    return sections


def group_by_section(elements: list[Document]) -> list[Document]:
    """
    Group consecutive elements under their nearest Title heading.
    Each section becomes one Document — the chunker then splits these into final chunks.
    """
    sections: list[Document] = []
    current_title = "Preamble"
    current_texts: list[str] = []
    # seed metadata from the very first element so even pre-title content has a source
    current_meta: dict = {
        k: v for k, v in (elements[0].metadata.items() if elements else {}.items())
        if k in ("filename", "page_number", "source")
    }

    for el in elements:
        category = el.metadata.get("category", "")
        text = el.page_content.strip()

        if category == "Title":
            # Save accumulated section before starting a new one
            if current_texts:
                sections.append(Document(
                    page_content="\n\n".join(current_texts),
                    metadata={**current_meta, "section_title": current_title},
                ))
            current_title = text
            current_texts = []
            current_meta = {
                k: v for k, v in el.metadata.items()
                if k in ("filename", "page_number", "source")
            }
        else:
            if len(text) >= MIN_ELEMENT_LENGTH:
                current_texts.append(text)

    # Flush last section
    if current_texts:
        sections.append(Document(
            page_content="\n\n".join(current_texts),
            metadata={**current_meta, "section_title": current_title},
        ))

    return sections


if __name__ == "__main__":
    docs = load_documents()
    for doc in docs[:3]:
        print("\n--- Section ---")
        print(f"Title  : {doc.metadata.get('section_title', '')}")
        print(f"Page   : {doc.metadata.get('page_number', '')}")
        print(f"Source : {doc.metadata.get('filename', '')}")
        print(f"Content: {doc.page_content[:300]}")
