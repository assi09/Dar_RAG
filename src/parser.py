import re
from pathlib import Path
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document


DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
MIN_ELEMENT_LENGTH = 20  # filter OCR noise / stray characters

# Section titles that are front matter — no useful answer content
FRONT_MATTER_TITLES = {
    "contents", "table of contents", "glossary", "acknowledgments",
    "acknowledgements", "acronyms", "abbreviations", "acronyms and abbreviations",
    "copyright", "preface", "foreword", "overview",
}

# OCR artifacts from CIS Controls safeguard table columns:
# - colored IG1/IG2/IG3 dot indicators → elele, jeleie, Grup, lelele
# - colored security function badges → t Protect ], t Detect ], t Respond ]
# - asset type column noise → |__vevices, | ona |, | GED |
_TABLE_NOISE = re.compile(
    r'\.?\s*(?:Protect|Detect|Respond|Identify)\s*\]\s*|'  # badge OCR: . Detect ], t Protect ]
    r't\s+(?:Protect|Detect|Respond|Identify)\s*\]|'
    r'@[A-Z]{2,}|'                                     # @EEEED, @ENED
    r'\b(?:Grup|GERD|GEED|EEED|ENED|GED|Le)\b|'       # known garbage words
    r'\b[jl]?ele(?:ie|le)?\b|'                         # elele, jele, jeleie, lelele
    r'\b__\w+|'                                        # __vevices
    r'\[\s*[a-z]{2,6}\s*\|*|'                          # [elele|, [jele
    r'\|\s{0,2}[a-z]{2,4}\s{0,2}\|'                   # | ona |, | eje |
)


def clean_text(text: str) -> str:
    """Remove OCR artifacts from table cells that corrupt safeguard entries."""
    text = _TABLE_NOISE.sub('', text)
    text = re.sub(r'\|+', ' ', text)          # stray pipe characters
    text = re.sub(r'[ \t]{2,}', ' ', text)    # collapse multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)    # collapse excess blank lines
    return text.strip()


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
            print(f"  WARNING: parse error ({type(e).__name__}), retrying hi_res...")
            elements = loader.load()  # retry once — Tesseract errors are often transient
        print(f"  -> {len(elements)} element(s) extracted")
        all_elements.extend(elements)

    sections = group_by_section(all_elements)
    print(f"\nTotal sections after grouping: {len(sections)}")
    return sections


import re as _re
_SAFEGUARD_RE = _re.compile(r'^\d+\.\d+[\s\.]')  # matches "13.1 ", "7.2 ", "12.3."


def _is_safeguard_title(text: str) -> bool:
    """True if this title is a numbered safeguard (e.g. '13.1 Centralize Security Event Alerting')
    rather than a top-level control heading (e.g. 'Control 13: Network Monitoring and Defense')."""
    return bool(_SAFEGUARD_RE.match(text.strip()))


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

        if category == "Title" and not _is_safeguard_title(text):
            # Save accumulated section before starting a new one
            # Skip front matter sections — they contain no answer-worthy content
            if current_texts and current_title.lower().strip() not in FRONT_MATTER_TITLES:
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
        elif category == "Title" and _is_safeguard_title(text):
            # Numbered safeguard titles (e.g. "13.1 Centralize Security Event Alerting")
            # are kept as content — NOT as section boundaries — so all safeguards for a
            # control stay together in one chunk instead of being split into micro-sections
            text_clean = clean_text(text)
            if len(text_clean) >= MIN_ELEMENT_LENGTH:
                current_texts.append(text_clean)
        else:
            text = clean_text(text)
            if len(text) >= MIN_ELEMENT_LENGTH:
                current_texts.append(text)

    # Flush last section
    if current_texts and current_title.lower().strip() not in FRONT_MATTER_TITLES:
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
