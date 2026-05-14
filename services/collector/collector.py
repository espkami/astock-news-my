"""
services/collector/collector.py
新闻采集服务 — 异步多源采集（财联社 / 东方财富 / 新浪财经 / RSS）
"""
from __future__ import annotations
import asyncio
import hashlib
import re
from datetime import datetime, timezone
from typing import AsyncIterator

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.models import RawNews
from shared.config import get_settings

settings = get_settings()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def make_news_id(title: str, source: str) -> str:
    return hashlib.sha256(f"{title}{source}".encode()).hexdigest()[:32]


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text()).strip()


# ─── 基类 ─────────────────────────────────────────────────────────────────────

class BaseCollector:
    source_name: str = "base"

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def fetch(self, url: str, **kwargs) -> str:
        async with self.session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15), **kwargs) as resp:
            resp.raise_for_status()
            return await resp.text()

    async def collect(self) -> AsyncIterator[RawNews]:
        raise NotImplementedError


# ─── 财联社采集器 ─────────────────────────────────────────────────────────────

class CLSCollector(BaseCollector):
    source_name = "cls"
    API_URL = "https://www.cls.cn/nodeapi/telegrams"

    async def collect(self) -> AsyncIterator[RawNews]:
        try:
            html = await self.fetch(self.API_URL, params={"app": "CLS", "os": "web", "sv": "8.4.6"})
            # 财联社电报接口返回JSON
            import orjson
            data = orjson.loads(html)
            items = data.get("data", {}).get("roll_data", [])
            for item in items[:20]:
                title = item.get("title", "") or item.get("brief", "")
                content = clean_text(item.get("content", title))
                if not title:
                    continue
                ts = item.get("ctime", 0)
                pub = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.utcnow()
                yield RawNews(
                    id=make_news_id(title, self.source_name),
                    title=title,
                    content=content,
                    source=self.source_name,
                    url=f"https://www.cls.cn/detail/{item.get('id', '')}",
                    published_at=pub,
                )
        except Exception as e:
            logger.warning(f"[CLS] 采集失败: {e}")


# ─── 东方财富采集器 ───────────────────────────────────────────────────────────

class EastMoneyCollector(BaseCollector):
    source_name = "eastmoney"
    # 东方财富快讯接口
    API_URL = (
        "https://push2ex.eastmoney.com/getTopicCount"
        "?type=1&pageindex=0&pagesize=20&t=1&_={ts}"
    )
    NEWS_API = "https://newsapi.eastmoney.com/kuaixun/v1/getlist_101_ajaxResult_50_1_.html"

    async def collect(self) -> AsyncIterator[RawNews]:
        try:
            html = await self.fetch(self.NEWS_API)
            data = __import__("orjson").loads(html)
            items = data.get("LiveList", [])
            for item in items[:20]:
                title = item.get("title", "")
                content = item.get("digest", title)
                if not title:
                    continue
                pub_str = item.get("showtime", "")
                try:
                    pub = datetime.strptime(pub_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    pub = datetime.utcnow()
                yield RawNews(
                    id=make_news_id(title, self.source_name),
                    title=title,
                    content=clean_text(content),
                    source=self.source_name,
                    url=item.get("url", ""),
                    published_at=pub,
                )
        except Exception as e:
            logger.warning(f"[EastMoney] 采集失败: {e}")


# ─── 新浪财经采集器 ───────────────────────────────────────────────────────────

class SinaCollector(BaseCollector):
    source_name = "sina"
    RSS_URL = "https://finance.sina.com.cn/roll/index.d.html?cateids=57,516&page=1"

    async def collect(self) -> AsyncIterator[RawNews]:
        try:
            html = await self.fetch(self.RSS_URL)
            soup = BeautifulSoup(html, "lxml")
            items = soup.select(".list_009 li a")[:20]
            for a in items:
                title = a.get_text(strip=True)
                url = a.get("href", "")
                if not title or not url:
                    continue
                yield RawNews(
                    id=make_news_id(title, self.source_name),
                    title=title,
                    content=title,   # 快讯只有标题，content填title
                    source=self.source_name,
                    url=url,
                    published_at=datetime.utcnow(),
                )
        except Exception as e:
            logger.warning(f"[Sina] 采集失败: {e}")


# ─── RSS 聚合采集器 ───────────────────────────────────────────────────────────

RSS_FEEDS = [
    ("reuters_cn",    "https://feeds.reuters.com/reuters/CNTopNews"),
    ("bloomberg_cn",  "https://feeds.bloomberg.com/markets/news.rss"),
    ("caixin",        "https://www.caixin.com/rss/all.xml"),
    ("yicai",         "https://www.yicai.com/rss/"),
]


class RSSCollector(BaseCollector):
    source_name = "rss"

    async def _parse_feed(self, name: str, url: str) -> list[RawNews]:
        results = []
        try:
            html = await self.fetch(url)
            feed = feedparser.parse(html)
            for entry in feed.entries[:10]:
                title = entry.get("title", "")
                content = clean_text(entry.get("summary", title))
                pub_parsed = entry.get("published_parsed")
                pub = datetime(*pub_parsed[:6], tzinfo=timezone.utc) if pub_parsed else datetime.utcnow()
                if not title:
                    continue
                results.append(RawNews(
                    id=make_news_id(title, name),
                    title=title,
                    content=content,
                    source=name,
                    url=entry.get("link", url),
                    published_at=pub,
                ))
        except Exception as e:
            logger.warning(f"[RSS:{name}] 采集失败: {e}")
        return results

    async def collect(self) -> AsyncIterator[RawNews]:
        tasks = [self._parse_feed(name, url) for name, url in RSS_FEEDS]
        results = await asyncio.gather(*tasks)
        for batch in results:
            for news in batch:
                yield news


# ─── 统一采集入口 ─────────────────────────────────────────────────────────────

class NewsCollector:
    """统一采集入口，管理所有采集器"""

    def __init__(self):
        self._seen: set[str] = set()  # 去重缓存

    async def collect_all(self) -> list[RawNews]:
        """并发采集所有源，去重后返回"""
        connector = aiohttp.TCPConnector(limit=20, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            collectors: list[BaseCollector] = []
            if settings.enable_cls:
                collectors.append(CLSCollector(session))
            if settings.enable_eastmoney:
                collectors.append(EastMoneyCollector(session))
            if settings.enable_sina:
                collectors.append(SinaCollector(session))
            if settings.enable_rss:
                collectors.append(RSSCollector(session))

            results: list[RawNews] = []
            for collector in collectors:
                async for news in collector.collect():
                    if news.id not in self._seen:
                        self._seen.add(news.id)
                        results.append(news)
                        logger.debug(f"采集: [{news.source}] {news.title[:40]}")

        logger.info(f"本次采集完成，共 {len(results)} 条新闻")
        return results
