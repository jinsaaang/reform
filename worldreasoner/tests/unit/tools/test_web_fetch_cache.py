"""Regression tests for process-local WebFetchTool reuse."""

import asyncio

from src.tools.collectors.web_fetch import WebFetchTool


def test_successful_fetch_is_reused_across_tool_instances(monkeypatch):
    calls = 0

    async def fake_fetch(self, url, timeout=15, timestamp=None):
        nonlocal calls
        calls += 1
        return {
            "url": url,
            "title": "Example",
            "markdown": "Reusable article content. " * 10,
            "metadata": {"datePublished": "2026-04-30"},
            "success": True,
        }

    WebFetchTool.clear_cache()
    monkeypatch.setattr(WebFetchTool, "_fetch_async", fake_fetch)

    first = WebFetchTool().forward("https://example.com/article")
    second = WebFetchTool().forward("https://example.com/article/")

    assert first.success is True
    assert second.content == first.content
    assert calls == 1


def test_failed_fetch_is_temporarily_cached(monkeypatch):
    calls = 0

    async def fake_fetch(self, url, timeout=15, timestamp=None):
        nonlocal calls
        calls += 1
        return {"url": url, "success": False, "error": "blocked"}

    WebFetchTool.clear_cache()
    monkeypatch.setattr(WebFetchTool, "_fetch_async", fake_fetch)

    WebFetchTool().forward("https://example.com/blocked")
    WebFetchTool().forward("https://example.com/blocked")

    assert calls == 1


def test_complete_fetch_path_has_a_hard_deadline(monkeypatch):
    async def never_finishes(self, url, timeout=15, timestamp=None):
        await asyncio.sleep(60)

    WebFetchTool.clear_cache()
    monkeypatch.setattr(WebFetchTool, "_fetch_async", never_finishes)
    monkeypatch.setattr(WebFetchTool, "_total_timeout_grace_seconds", 0.01)

    result = WebFetchTool().forward("https://example.com/hung", timeout=0)

    assert result.success is False
    assert "total deadline" in result.error
