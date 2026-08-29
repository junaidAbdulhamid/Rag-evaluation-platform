"""Document loading.

Phase 1 supports plain-text formats only: `.txt` and `.md`. The loader's job is
narrow on purpose - read bytes, wrap them in a `Document` - so that swapping in a
PDF/HTML parser later is a change in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

from app.models import Document

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_document(path: str | Path) -> Document:
    """Load a single file into a `Document`.

    The `document_id` is the filename without extension (e.g. `refund_policy.md` ->
    `refund_policy`). This is what the evaluation dataset in Phase 2 refers to in its
    `relevant_document_ids` field, so it must be stable and human-readable.
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {path.suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    text = path.read_text(encoding="utf-8")
    return Document(
        document_id=path.stem,
        filename=path.name,
        text=text,
        metadata={"path": str(path), "suffix": path.suffix.lower()},
    )


def load_documents(directory: str | Path) -> list[Document]:
    """Load every supported file in `directory` (non-recursive), sorted by filename.

    Sorting makes ingestion deterministic, which matters once we start comparing
    experiment runs - the same corpus should always produce the same chunk ids.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    documents = [
        load_document(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]

    if not documents:
        raise FileNotFoundError(
            f"No {sorted(SUPPORTED_SUFFIXES)} files found in {directory}"
        )
    return documents
