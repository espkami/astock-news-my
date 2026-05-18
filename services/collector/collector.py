"""
services/collector/collector.py
新闻采集服务 — 使用国内可访问的直连 RSS 源

问题根因：原版依赖 Google News RSS（news.google.com），国内服务器无法访问。
本版改用经过验证可在国内直连的 RSS 数据源。
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import AsyncIterator

import aiohttp
import feedparser
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

RSS_HEADERS = {
    "User-Agent": CHROME_UA,
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

FUTURE_TOLERANCE = timedelta(hours=1)
MAX_NEWS_AGE     = timedelta(hours=96)
ITEMS_PER_FEED   = 10

# ── 国内可直连 RSS 数据源 ──────────────────────────────────────────────────────
#
# 来源均为国内服务器，经过测试可以直接访问：
# - 新浪财经 RSS（多个分类频道）
# - 网易财经 RSS
# - 凤凰财经 RSS
# - 中国证券网（上证报）RSS
# - 证券时报 RSS
# - 央视财经 RSS
# - 新华财经 RSS
# - 华尔街见闻 RSS
#
ASTOCK_FEEDS: list[tuple[str, str]] = [

    # ── 新浪财经（多个 RSS 频道）─────────────────────────────────────────────
    ("新浪财经-股市",    "https://feed.sina.com.cn/api/roll/get?pageid=155&lid=2509&num=20&versionNumber=1.2.5&page=1&encode=utf-8&callback=feedCallback"),
    ("新浪财经-要闻",    "https://feed.sina.com.cn/api/roll/get?pageid=155&lid=2516&num=20&versionNumber=1.2.5&page=1&encode=utf-8"),
    ("新浪财经-公司",    "https://feed.sina.com.cn/api/roll/get?pageid=155&lid=2515&num=20&versionNumber=1.2.5&page=1&encode=utf-8"),
    ("新浪财经-宏观",    "https://feed.sina.com.cn/api/roll/get?pageid=155&lid=2512&num=20&versionNumber=1.2.5&page=1&encode=utf-8"),

    # ── 网易财经 RSS ─────────────────────────────────────────────────────────
    ("网易财经",         "https://money.163.com/special/00251LOP/rss_newstock.xml"),
    ("网易股票",         "https://money.163.com/special/00251LOP/rss_gupiao.xml"),

    # ── 央视财经 RSS ─────────────────────────────────────────────────────────
    ("央视财经",         "https://rss.cctv.com/2006/07/25/ARTI1232380038863458.xml"),

    # ── 中国证券网（上海证券报）RSS ──────────────────────────────────────────
    ("上证报-要闻",      "https://www.cnstock.com/rss/index.xml"),

    # ── 证券时报 RSS ─────────────────────────────────────────────────────────
    ("证券时报",         "https://www.stcn.com/rss.html"),

    # ── 新华网财经 RSS ───────────────────────────────────────────────────────
    ("新华财经",         "http://www.xinhuanet.com/fortune/index.rss"),

    # ── 凤凰财经 RSS ─────────────────────────────────────────────────────────
    ("凤凰财经",         "https://finance.ifeng.com/rss/index.xml"),

    # ── 华尔街见闻 RSS ───────────────────────────────────────────────────────
    ("华尔街见闻",       "https://wallstreetcn.com/rss.xml"),

    # ── 东方财富网 RSS ───────────────────────────────────────────────────────
    ("东方财富-要闻",    "https://rssfeed.eastmoney.com/rss/news"),
    ("东方财富-股票",    "https://rssfeed.eastmoney.com/rss/gupiao"),

    # ── 财经网 RSS ───────────────────────────────────────────────────────────
    ("财经网",           "https://www.caijing.com.cn/rss/all.xml"),

    # ── 21世纪经济报道 RSS ───────────────────────────────────────────────────
    ("21世纪经济",       "https://www.21jingji.com/tools/getRss.do"),
]


# ── A 股关键词分类器 ─────────────────────────────────────────────────────────

CRITICAL_KEYWORDS = {
    "熔断": "economic", "股市崩盘": "economic", "流动性危机": "economic",
    "金融危机": "economic", "银行破产": "economic", "系统性风险": "economic",
}

HIGH_KEYWORDS = {
    "大跌": "economic", "暴跌": "economic", "跌停": "economic",
    "大涨": "economic", "涨停": "economic", "暴涨": "economic",
    "降准": "economic", "降息": "economic", "加息": "economic",
    "制裁": "economic", "贸易战": "economic", "关税": "economic",
    "退市": "economic", "强制退市": "economic",
    "重大重组": "economic", "借壳上市": "economic",
}

MEDIUM_KEYWORDS = {
    "业绩预增": "economic", "业绩预减": "economic", "业绩爆雷": "economic",
    "定增": "economic", "回购": "economic", "分红": "economic",
    "并购": "economic", "重组": "economic", "股权转让": "economic",
    "北向资金": "economic", "主力资金": "economic",
    "政策利好": "economic", "政策利空": "economic",
    "资金流入": "economic",
}

LOW_KEYWORDS = {
    "季报": "economic", "年报": "economic", "中报": "economic",
    "股东大会": "economic", "高管变动": "economic",
    "新产品": "tech", "研发投入": "tech",
    "市值": "economic", "估值": "economic",
}


def classify_astock(title: str) -> tuple[str, str, float]:
    for kw, cat in CRITICAL_KEYWORDS.items():
        if kw in title:
            return ("critical", cat, 0.9)
    for kw, cat in HIGH_KEYWORDS.items():
        if kw in title:
            return ("high", cat, 0.8)
    for kw, cat in MEDIUM_KEYWORDS.items():
        if kw in title:
            return ("medium", cat, 0.7)
    for kw, cat in LOW_KEYWORDS.items():
        if kw in title:
            return ("low", cat, 0.6)
    return ("info", "general", 0.3)


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def make_news_id(title: str, source: str) -> str:
    return hashlib.sha256(f"{title}{source}".encode()).hexdigest()[:32]


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text()).strip()


def looks_like_rss(text: str) -> bool:
    head = text[:2048].lower()
    if re.search(r"<!doctype\s+html|<html[\s>]", head):
        return False
    return bool(re.search(r"<rss[\s>]|<feed[\s>]|<rdf:rdf[\s>]", head))


def parse_pubdate(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
    except Exception:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    if dt > now + FUTURE_TOLERANCE:
        return None
    return dt


def parse_sina_json(text: str, source: str) -> list[RawNews]:
    """
    解析新浪财经 JSON API 响应（非标准 RSS）。
    格式：feedCallback({...}) 或直接 JSON。
    """
    import json
    results = []
    # 去掉 JSONP 包装
    text = re.sub(r"^[a-zA-Z_]\w*\(", "", text).rstrip(");")
    try:
        data = json.loads(text)
    except Exception:
        return results

    items = data.get("result", {}).get("data", [])
    now = datetime.now(tz=timezone.utc)
    cutoff = now - MAX_NEWS_AGE

    for item in items[:ITEMS_PER_FEED]:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        intro = item.get("intro") or item.get("summary") or title
        url   = item.get("url") or item.get("link") or ""

        # 时间解析
        ctime = item.get("ctime") or item.get("create_time") or ""
        pub: datetime | None = None
        if ctime:
            try:
                ts = int(ctime)
                pub = datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pub = parse_pubdate(str(ctime))

        if pub is None:
            pub = now
        if pub < cutoff:
            continue

        results.append(RawNews(
            id=make_news_id(title, source),
            title=title,
            content=clean_text(intro) if "<" in intro else intro,
            source=source,
            url=url,
            published_at=pub,
        ))
    return results


# ── RSS 采集器 ───────────────────────────────────────────────────────────────

class RSSCollector:

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _fetch(self, url: str) -> str | None:
        try:
            async with self.session.get(
                url,
                headers=RSS_HEADERS,
                timeout=aiohttp.ClientTimeout(total=15),
                ssl=False,
            ) as resp:
                if not resp.ok:
                    logger.debug(f"[RSS] HTTP {resp.status}: {url[:60]}")
                    return None
                return await resp.text()
        except Exception as e:
            logger.debug(f"[RSS] 拉取失败: {url[:60]} — {e}")
            return None

    async def parse_feed(self, name: str, url: str) -> list[RawNews]:
        text = await self._fetch(url)
        if not text:
            logger.warning(f"[{name}] 无法获取内容")
            return []

        # 新浪财经 JSON API
        if "sina.com.cn/api" in url:
            results = parse_sina_json(text, name)
            if results:
                logger.info(f"[{name}] 采集 {len(results)} 条")
            else:
                logger.warning(f"[{name}] JSON 解析为空")
            return results

        # 标准 RSS/Atom
        if not looks_like_rss(text):
            logger.warning(f"[{name}] 响应非 RSS（可能被拦截或改版）: {url[:60]}")
            return []

        feed = feedparser.parse(text)
        now    = datetime.now(tz=timezone.utc)
        cutoff = now - MAX_NEWS_AGE
        results: list[RawNews] = []

        for entry in feed.entries[:ITEMS_PER_FEED]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue

            # 日期解析
            pub: datetime | None = None
            pub_parsed = entry.get("published_parsed")
            if pub_parsed:
                try:
                    pub = datetime(*pub_parsed[:6], tzinfo=timezone.utc)
                    if pub > now + FUTURE_TOLERANCE:
                        continue
                except Exception:
                    pub = None

            if pub is None:
                raw = entry.get("published", "") or entry.get("updated", "")
                pub = parse_pubdate(raw)

            # 无日期：使用当前时间（宽松处理国内部分 RSS 无日期问题）
            if pub is None:
                pub = now

            if pub < cutoff:
                continue

            summary = entry.get("summary", "") or ""
            content = clean_text(summary) if summary else title
            link    = entry.get("link", url)

            results.append(RawNews(
                id=make_news_id(title, name),
                title=title,
                content=content,
                source=name,
                url=link,
                published_at=pub,
            ))

        if results:
            logger.info(f"[{name}] 采集 {len(results)} 条")
        else:
            logger.warning(f"[{name}] 解析为空（共 {len(feed.entries)} 条 entries）")
        return results

    async def collect_all(self) -> list[RawNews]:
        BATCH_SIZE = 5
        all_results: list[RawNews] = []

        for i in range(0, len(ASTOCK_FEEDS), BATCH_SIZE):
            batch = ASTOCK_FEEDS[i:i + BATCH_SIZE]
            tasks = [self.parse_feed(name, url) for name, url in batch]
            settled = await asyncio.gather(*tasks, return_exceptions=True)
            for result in settled:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"[RSS] batch 异常: {result}")

        return all_results


# ── 统一采集入口 ─────────────────────────────────────────────────────────────

class NewsCollector:

    def __init__(self):
        self._seen: set[str] = set()

    async def collect_all(self) -> list[RawNews]:
        connector = aiohttp.TCPConnector(limit=20, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            collector = RSSCollector(session)
            raw = await collector.collect_all()

        results: list[RawNews] = []
        for news in sorted(raw, key=lambda n: n.published_at, reverse=True):
            if news.id not in self._seen:
                self._seen.add(news.id)
                results.append(news)

        logger.info(f"本次采集完成，共 {len(results)} 条新闻（{len(ASTOCK_FEEDS)} 个 feed）")
        return results
