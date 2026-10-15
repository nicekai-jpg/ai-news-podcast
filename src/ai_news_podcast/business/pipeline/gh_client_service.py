"""GitHub REST API 轻量客户端（项目雷达专用）。

只封装雷达需要的两个端点：仓库搜索与 README 读取。
未显式传 token 时读取 GITHUB_TOKEN 环境变量；都缺失则匿名访问
（限流更紧，但每日一次的扫描量足够）。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

_GITHUB_API = "https://api.github.com"
_README_MAX_CHARS = 40000


class GhClient:
    """异步 GitHub REST 客户端。测试时传入 MockTransport 构造的 AsyncClient。

    接管传入 client 的生命周期（`aclose()` 会关闭它）；超时等传输参数由构造方设置。
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        token: str | None = None,
        sleep_seconds: float = 1.0,
    ) -> None:
        self._client = client
        self._sleep_seconds = sleep_seconds
        headers = {"Accept": "application/vnd.github+json"}
        resolved = token if token is not None else os.environ.get("GITHUB_TOKEN") or ""
        if resolved:
            headers["Authorization"] = f"Bearer {resolved}"
        self._headers = headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search_repos(
        self, query: str, *, sort: str = "stars", per_page: int = 30
    ) -> list[dict[str, Any]]:
        params = {"q": query, "sort": sort, "order": "desc", "per_page": per_page}
        resp = await self._client.get(
            f"{_GITHUB_API}/search/repositories", params=params, headers=self._headers
        )
        resp.raise_for_status()
        # 调用间隔 1s；匿名 search 限流 10 次/分钟，勿高频调用
        await asyncio.sleep(self._sleep_seconds)
        return list(resp.json().get("items", []))

    async def fetch_readme_text(self, repo: str) -> str:
        """返回 README 原文（截断），404 视为无 README 返回空串。"""
        headers = dict(self._headers)
        headers["Accept"] = "application/vnd.github.raw+json"
        resp = await self._client.get(f"{_GITHUB_API}/repos/{repo}/readme", headers=headers)
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        return resp.text[:_README_MAX_CHARS]

    async def fetch_repo(self, full_name: str) -> dict[str, Any] | None:
        """GET /repos/{full_name}:单仓库元数据(stars/topics/description 等)。

        404(仓库不存在/已删除)返回 None;其余错误上抛,由调用方决定降级。
        """
        resp = await self._client.get(f"{_GITHUB_API}/repos/{full_name}", headers=self._headers)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    async def fetch_text(self, url: str) -> str:
        """通用 HTTP GET 文本(供 Trending RSS 等榜单镜像源使用)。

        带浏览器 UA:公共 RSSHub 实例对无 UA 的脚本请求会 403。
        """
        headers = dict(self._headers)
        headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        resp = await self._client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text
