import json
from pathlib import Path

import pytest

from src import arxiv_downloader


class FakeResponse:
    headers = {"content-type": "application/pdf"}
    content = b"%PDF fake"

    def raise_for_status(self) -> None:
        return None


def test_retry_with_backoff_reports_attempt_summary(monkeypatch) -> None:
    monkeypatch.setattr(arxiv_downloader.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        arxiv_downloader.retry_with_backoff(
            lambda: (_ for _ in ()).throw(ConnectionError("network down")),
            attempts=3,
            sleep_seconds=0,
            operation_name="arXiv search",
        )


def test_download_pdf_retries_then_writes_file(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}

    def fake_get(_url: str, timeout: int):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionError("temporary failure")
        return FakeResponse()

    monkeypatch.setattr(arxiv_downloader.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_downloader.time, "sleep", lambda _seconds: None)

    output_path = tmp_path / "paper.pdf"
    arxiv_downloader.download_pdf("https://arxiv.org/pdf/1234.5678", output_path, retries=2, sleep_seconds=0)

    assert output_path.read_bytes() == b"%PDF fake"
    assert calls["count"] == 2


def test_load_paper_ids_reads_regression_json(tmp_path: Path) -> None:
    id_file = tmp_path / "papers.json"
    id_file.write_text(
        json.dumps(
            [
                {"arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
                {"paper_id": "1810.04805"},
                "2005.11401",
            ]
        ),
        encoding="utf-8",
    )

    assert arxiv_downloader.load_paper_ids(id_file) == ["1706.03762", "1810.04805", "2005.11401"]


def test_search_and_download_accepts_fixed_arxiv_ids(tmp_path: Path, monkeypatch) -> None:
    class FakeResult:
        title = "Fixed Paper"
        pdf_url = "https://example.com/fixed.pdf"
        authors = []
        summary = "summary"
        categories = []
        entry_id = "https://arxiv.org/abs/1706.03762"
        published = updated = type("Date", (), {"isoformat": lambda self: "2017-06-01T00:00:00"})()

        def get_short_id(self) -> str:
            return "1706.03762"

    calls = {}

    def fake_search_by_ids(paper_ids, **_kwargs):
        calls["paper_ids"] = paper_ids
        return [FakeResult()]

    def fake_download(_url: str, output_path: Path, **_kwargs) -> None:
        output_path.write_bytes(b"%PDF fake")

    monkeypatch.setattr(arxiv_downloader, "search_arxiv_by_ids", fake_search_by_ids)
    monkeypatch.setattr(arxiv_downloader, "download_pdf", fake_download)
    monkeypatch.setattr(arxiv_downloader.time, "sleep", lambda _seconds: None)

    papers = arxiv_downloader.search_and_download(
        output_dir=tmp_path,
        paper_ids=["1706.03762"],
        sleep_seconds=0,
    )

    assert calls["paper_ids"] == ["1706.03762"]
    assert papers[0].arxiv_id == "1706.03762"


def test_search_and_download_records_failed_pdf_download(tmp_path: Path, monkeypatch) -> None:
    class FakeResult:
        title = "A Paper"
        pdf_url = "https://example.com/a.pdf"
        authors = []
        summary = "summary"
        categories = []
        entry_id = "https://arxiv.org/abs/1"
        published = updated = type("Date", (), {"isoformat": lambda self: "2024-01-01T00:00:00"})()

        def get_short_id(self) -> str:
            return "2401.00001"

    monkeypatch.setattr(arxiv_downloader, "search_arxiv", lambda **_kwargs: [FakeResult()])
    monkeypatch.setattr(
        arxiv_downloader,
        "download_pdf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError("download failed")),
    )

    papers = arxiv_downloader.search_and_download("graph", output_dir=tmp_path, sleep_seconds=0)

    assert papers == []
    failure_text = (tmp_path / "metadata" / "arxiv_download_failures.json").read_text(encoding="utf-8")
    assert "download failed" in failure_text
