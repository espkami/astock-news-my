"""
services/collector/collector.py
新闻采集服务 — 2026年5月验证可用的数据源

可用源（共5个）：
  1. 财联社电报      https://www.cls.cn/nodeapi/updateTelegraphList
  2. 同花顺快讯      https://news.10jqka.com.cn/tapp/news/push/stock/
  3. 新浪财经滚动    https://feed.sina.com.cn/api/roll/get (pageid=153)
  4. 华尔街见闻快讯  https://api.wallstreetcn.com/apiv1/content/articles
  5. 华尔街见闻要闻  https://api.wallstreetcn.com/apiv1/content/lives
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from typing import List

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.models import RawNews
from shared.config import get_settings

settings = get_settings()

CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
BASE_HEADERS = {
    "User-Agent": CHROME_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

MAX_NEWS_AGE   = timedelta(hours=96)
FUTURE_TOLE    = timedelta(hours=1)
ITEMS_PER_SRC  = 20


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def make_id(title: str, source: str) -> str:
    return hashlib.sha256(f"{title}{source}".encode()).hexdigest()[:32]

def clean_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\s+", " ", soup.get_text()).strip()

def ts_to_dt(ts) -> datetime:
    """Unix 时间戳（秒）→ UTC datetime"""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except Exception:
        return datetime.now(tz=timezone.utc)

def is_valid_dt(dt: datetime) -> bool:
    now = datetime.now(tz=timezone.utc)
    return (now - MAX_NEWS_AGE) <= dt <= (now + FUTURE_TOLE)


# ── 各数据源采集函数 ─────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
async def _get(session: aiohttp.ClientSession, url: str, **kwargs) -> str:
    async with session.get(
        url,
        headers=BASE_HEADERS,
        timeout=aiohttp.ClientTimeout(total=12),
        ssl=False,
        **kwargs,
    ) as resp:
        resp.raise_for_status()
        return await resp.text(errors="replace")


async def fetch_cls(session: aiohttp.ClientSession) -> List[RawNews]:
    """财联社电报"""
    SOURCE = "财联社"
    url = (
        "https://www.cls.cn/nodeapi/updateTelegraphList"
        "?app=CLS&os=web&sv=7.7.5&hasFirstVipArticle=1&refresh=1&rn=20"
    )
    try:
        text = await _get(session, url)
        data = json.loads(text)
        items = data.get("data", {}).get("roll_data", [])
    except Exception as e:
        logger.warning(f"[{SOURCE}] 获取失败: {e}")
        return []

    results = []
    for it in items[:ITEMS_PER_SRC]:
        content_raw = it.get("content") or it.get("brief") or ""
        title = clean_html(content_raw)[:100] or "财联社电报"
        content = clean_html(content_raw)
        url_item = it.get("share_url") or it.get("jump_url") or ""
        pub_dt = ts_to_dt(it.get("ctime") or it.get("modified_time") or 0)
        if not is_valid_dt(pub_dt):
            continue
        results.append(RawNews(
            id=make_id(title, SOURCE),
            title=title,
            content=content,
            source=SOURCE,
            url=url_item,
            published_at=pub_dt,
        ))
    logger.info(f"[{SOURCE}] 采集 {len(results)} 条")
    return results


async def fetch_ths(session: aiohttp.ClientSession) -> List[RawNews]:
    """同花顺快讯"""
    SOURCE = "同花顺"
    url = "https://news.10jqka.com.cn/tapp/news/push/stock/?page=1&tag=&track=website&pagesize=20"
    try:
        text = await _get(session, url)
        clean = re.sub(r"^[a-zA-Z_$]\w*\(", "", text.strip()).rstrip(");")
        data = json.loads(clean)
        items = data.get("data", {}).get("list", [])
    except Exception as e:
        logger.warning(f"[{SOURCE}] 获取失败: {e}")
        return []

    results = []
    for it in items[:ITEMS_PER_SRC]:
        title = (it.get("title") or "").strip()
        if not title:
            continue
        content = clean_html(it.get("digest") or it.get("content") or title)
        url_item = it.get("url") or it.get("link") or ""
        pub_dt = ts_to_dt(it.get("time") or it.get("ctime") or 0)
        if not is_valid_dt(pub_dt):
            continue
        results.append(RawNews(
            id=make_id(title, SOURCE),
            title=title,
            content=content,
            source=SOURCE,
            url=url_item,
            published_at=pub_dt,
        ))
    logger.info(f"[{SOURCE}] 采集 {len(results)} 条")
    return results


async def fetch_sina(session: aiohttp.ClientSession) -> List[RawNews]:
    """新浪财经滚动新闻（pageid=153 可用）"""
    SOURCE = "新浪财经"
    results = []
    lids = ["2509", "2516", "2515", "2512"]  # 股市/要闻/公司/宏观

    for lid in lids:
        url = (
            f"https://feed.sina.com.cn/api/roll/get"
            f"?pageid=153&lid={lid}&num=20&versionNumber=1.2.5&page=1&encode=utf-8"
        )
        try:
            text = await _get(session, url)
            clean = re.sub(r"^[a-zA-Z_]\w*\(", "", text).rstrip(");")
            data = json.loads(clean)
            items = data.get("result", {}).get("data", [])
        except Exception as e:
            logger.debug(f"[{SOURCE}] lid={lid} 失败: {e}")
            continue

        for it in items[:ITEMS_PER_SRC]:
            title = (it.get("title") or "").strip()
            if not title:
                continue
            intro = it.get("intro") or it.get("summary") or title
            content = clean_html(intro) if "<" in intro else intro
            url_item = it.get("url") or it.get("wapurl") or ""
            pub_dt = ts_to_dt(it.get("ctime") or it.get("mtime") or 0)
            if not is_valid_dt(pub_dt):
                continue
            results.append(RawNews(
                id=make_id(title, SOURCE),
                title=title,
                content=content,
                source=SOURCE,
                url=url_item,
                published_at=pub_dt,
            ))

    logger.info(f"[{SOURCE}] 采集 {len(results)} 条")
    return results


async def fetch_wsj_articles(session: aiohttp.ClientSession) -> List[RawNews]:
    """华尔街见闻 — 文章快讯"""
    SOURCE = "华尔街见闻"
    url = "https://api.wallstreetcn.com/apiv1/content/articles?channel=global-channel&accept=article&limit=20"
    try:
        text = await _get(session, url)
        data = json.loads(text)
        items = data.get("data", {}).get("items", [])
    except Exception as e:
        logger.warning(f"[{SOURCE}-文章] 获取失败: {e}")
        return []

    results = []
    for it in items[:ITEMS_PER_SRC]:
        title = (it.get("title") or "").strip()
        if not title:
            continue
        content = clean_html(it.get("content_short") or it.get("subtitle") or title)
        uri = it.get("uri") or ""
        url_item = f"https://wallstreetcn.com/articles/{uri}" if uri else ""
        pub_dt = ts_to_dt(it.get("display_time") or it.get("published_at") or 0)
        if not is_valid_dt(pub_dt):
            continue
        results.append(RawNews(
            id=make_id(title, SOURCE),
            title=title,
            content=content,
            source=SOURCE,
            url=url_item,
            published_at=pub_dt,
        ))
    logger.info(f"[{SOURCE}-文章] 采集 {len(results)} 条")
    return results


async def fetch_wsj_lives(session: aiohttp.ClientSession) -> List[RawNews]:
    """华尔街见闻 — 快讯直播"""
    SOURCE = "华尔街见闻快讯"
    url = "https://api.wallstreetcn.com/apiv1/content/lives?channel=global-channel&limit=20"
    try:
        text = await _get(session, url)
        data = json.loads(text)
        items = data.get("data", {}).get("items", [])
    except Exception as e:
        logger.warning(f"[{SOURCE}] 获取失败: {e}")
        return []

    results = []
    for it in items[:ITEMS_PER_SRC]:
        raw_content = it.get("content_text") or it.get("content") or ""
        content = clean_html(raw_content)
        title = content[:80] or "华尔街见闻快讯"
        pub_dt = ts_to_dt(it.get("display_time") or 0)
        if not is_valid_dt(pub_dt):
            continue
        results.append(RawNews(
            id=make_id(title, SOURCE),
            title=title,
            content=content,
            source=SOURCE,
            url="https://wallstreetcn.com/live",
            published_at=pub_dt,
        ))
    logger.info(f"[{SOURCE}] 采集 {len(results)} 条")
    return results


# ── 统一采集入口 ─────────────────────────────────────────────────────────────

class NewsCollector:

    def __init__(self):
        self._seen: set[str] = set()

    async def collect_all(self) -> List[RawNews]:
        connector = aiohttp.TCPConnector(limit=20, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                fetch_cls(session),
                fetch_ths(session),
                fetch_sina(session),
                fetch_wsj_articles(session),
                fetch_wsj_lives(session),
            ]
            settled = await asyncio.gather(*tasks, return_exceptions=True)

        raw: List[RawNews] = []
        for result in settled:
            if isinstance(result, list):
                raw.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"[collector] 子任务异常: {result}")

        # 去重 + 时间排序
        results: List[RawNews] = []
        for news in sorted(raw, key=lambda n: n.published_at, reverse=True):
            if news.id not in self._seen:
                self._seen.add(news.id)
                results.append(news)

        logger.info(f"本次采集完成，共 {len(results)} 条（5个源）")
        return results
