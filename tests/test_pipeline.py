from pathlib import Path

from lending_agent.ingestion.pipeline import ingest_document, ingest_documents, resolve_input_paths


class TestIngestDocument:
    def test_pdf_produces_cited_chunks(self, sample_pdf_path):
        chunks = ingest_document(sample_pdf_path, chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 0
        assert all(c.source_file == "sample_policy.pdf" for c in chunks)
        assert all(c.page_number in (1, 2) for c in chunks)
        assert any("500,000" in c.text for c in chunks)

    def test_docx_produces_cited_chunks_without_page_numbers(self, sample_docx_path):
        chunks = ingest_document(sample_docx_path, chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 0
        assert all(c.source_file == "sample_policy.docx" for c in chunks)
        assert all(c.page_number is None for c in chunks)

    def test_chunk_ids_are_unique(self, sample_pdf_path):
        chunks = ingest_document(sample_pdf_path, chunk_size=200, chunk_overlap=30)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_to_dict_round_trips_cleanly(self, sample_docx_path):
        chunks = ingest_document(sample_docx_path, chunk_size=300, chunk_overlap=50)
        d = chunks[0].to_dict()
        assert d["source_file"] == "sample_policy.docx"
        assert d["text"] == chunks[0].text


class TestResolveInputPaths:
    def test_individual_files_pass_through(self, sample_pdf_path, sample_docx_path):
        resolved = resolve_input_paths([sample_pdf_path, sample_docx_path])
        assert set(resolved) == {sample_pdf_path, sample_docx_path}

    def test_folder_is_expanded_to_supported_files(self, tmp_path, sample_pdf_path, sample_docx_path):
        import shutil

        folder = tmp_path / "policies"
        folder.mkdir()
        shutil.copy(sample_pdf_path, folder / "a.pdf")
        shutil.copy(sample_docx_path, folder / "b.docx")
        (folder / "notes.txt").write_text("not a policy doc")  # should be ignored

        resolved = resolve_input_paths([str(folder)])
        names = {Path(p).name for p in resolved}
        assert names == {"a.pdf", "b.docx"}

    def test_missing_path_raises(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            resolve_input_paths(["/no/such/file.pdf"])

    def test_empty_folder_raises(self, tmp_path):
        import pytest

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            resolve_input_paths([str(empty)])

    def test_duplicate_paths_are_deduplicated(self, sample_pdf_path):
        resolved = resolve_input_paths([sample_pdf_path, sample_pdf_path])
        assert resolved == [sample_pdf_path]


class TestIngestDocuments:
    def test_ingests_multiple_files_keyed_by_path(self, sample_pdf_path, sample_docx_path):
        results = ingest_documents([sample_pdf_path, sample_docx_path], chunk_size=300, chunk_overlap=50)
        assert set(results.keys()) == {sample_pdf_path, sample_docx_path}
        assert all(len(chunks) > 0 for chunks in results.values())

    def test_chunk_ids_stay_unique_across_files_with_same_basename(self, tmp_path, sample_pdf_path):
        import shutil

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        shutil.copy(sample_pdf_path, dir_a / "policy.pdf")
        shutil.copy(sample_pdf_path, dir_b / "policy.pdf")  # same filename, different file

        results = ingest_documents([str(dir_a / "policy.pdf"), str(dir_b / "policy.pdf")])
        all_ids = [c.chunk_id for chunks in results.values() for c in chunks]
        assert len(all_ids) == len(set(all_ids))

    def test_ingests_a_whole_folder(self, tmp_path, sample_pdf_path, sample_docx_path):
        import shutil

        folder = tmp_path / "batch"
        folder.mkdir()
        shutil.copy(sample_pdf_path, folder / "one.pdf")
        shutil.copy(sample_docx_path, folder / "two.docx")

        results = ingest_documents([str(folder)])
        assert len(results) == 2
        total_chunks = sum(len(c) for c in results.values())
        assert total_chunks > 0


