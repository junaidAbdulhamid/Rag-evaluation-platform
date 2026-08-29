from pathlib import Path

import pytest

from app.ingestion.loader import load_document, load_documents


def test_load_document_uses_filename_stem_as_id(tmp_path: Path):
    path = tmp_path / "refund_policy.md"
    path.write_text("Refunds within 30 days.", encoding="utf-8")

    doc = load_document(path)

    assert doc.document_id == "refund_policy"
    assert doc.filename == "refund_policy.md"
    assert doc.text == "Refunds within 30 days."
    assert doc.metadata["suffix"] == ".md"


def test_load_document_rejects_unsupported_type(tmp_path: Path):
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(path)


def test_load_documents_reads_txt_and_md_sorted_and_ignores_others(tmp_path: Path):
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert [d.document_id for d in docs] == ["a", "b"]  # sorted, json skipped


def test_load_documents_raises_when_directory_has_no_supported_files(tmp_path: Path):
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


def test_load_documents_raises_on_missing_directory(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        load_documents(tmp_path / "does_not_exist")
