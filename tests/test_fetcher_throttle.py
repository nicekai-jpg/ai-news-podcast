"""Tests for fetcher domain throttling."""

from __future__ import annotations

import asyncio

import pytest

from ai_news_podcast.pipeline.fetcher import _DomainThrottle


class TestDomainThrottle:
    @pytest.mark.asyncio
    async def test_same_domain_waits(self) -> None:
        throttle = _DomainThrottle(interval=0.1)
        t0 = asyncio.get_event_loop().time()
        await throttle.wait("example.com")
        await throttle.wait("example.com")
        t1 = asyncio.get_event_loop().time()
        assert t1 - t0 >= 0.1

    @pytest.mark.asyncio
    async def test_different_domains_no_wait(self) -> None:
        throttle = _DomainThrottle(interval=1.0)
        t0 = asyncio.get_event_loop().time()
        await throttle.wait("a.com")
        await throttle.wait("b.com")
        t1 = asyncio.get_event_loop().time()
        assert t1 - t0 < 0.5  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_zero_interval(self) -> None:
        throttle = _DomainThrottle(interval=0.0)
        t0 = asyncio.get_event_loop().time()
        await throttle.wait("example.com")
        await throttle.wait("example.com")
        t1 = asyncio.get_event_loop().time()
        assert t1 - t0 < 0.1
