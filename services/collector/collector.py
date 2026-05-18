"""
services/collector/collector.py
新闻采集服务 — 异步多源采集（财联社 / 东方财富 / 新浪财经 / RSS）

修复说明（参考 worldmonitor/server/worldmonitor/news/v1/ 实现）：
1. 财联社 API URL 已更新，增加必要请求头
2. 东方财富改用可用的快讯接口
3. 新浪财经改用 JSON API，不再依赖易变的 HTML 结构
4. RSS 源全部换成当前可用地址，并新增 A 股专项 Google News RSS
5. 增加 HTML 嗅探过滤（同 worldmonitor looksLikeRssXml），避免 Cloudflare 拦截页进缓存
6. 增加严格的日期校验，拒绝未来时间戳和无时间戳条目（同 worldmonitor U2/R2）
7. 网络请求增加更完整的浏览器 User-Agent 和 Accept-Language 头
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import AsyncIterator
import urllib.parse

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.models import RawNews
from shared.config import get_settings

settings = get_settings()

# ── 请求头（模拟 Chrome，参考 worldmonitor CHROME_UA） ─────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 未来时间宽限（同 worldmonitor FUTURE_DATE_TOLERANCE_MS = 1h）
FUTURE_TOLERANCE = timedelta(hours=1)


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def make_news_id(title: str, source: str) -> str:
    return hashlib.sha256(f"{title}{source}".encode()).hexdigest()[:32]


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text()).strip()


def looks_like_rss(text: str) -> bool:
    """
    嗅探响应体是否为真实 RSS/Atom/RDF，过滤掉 Cloudflare 拦截页等 HTML 墙。
    逻辑同 worldmonitor looksLikeRssXml。
    """
    head = text[:2048].lower()
    if re.search(r"<!doctype\s+html|<html[\s>]", head):
        return False
    return bool(re.search(r"<rss[\s>]|<feed[\s>]|<rdf:rdf[\s>]", head))


def parse_pubdate(date_str: str) -> datetime | None:
    """
    解析 RSS pubDate / Atom published 等日期字符串，返回 aware datetime。
    - 拒绝未来时间（宽限 1h）
    - 解析失败返回 None（由调用方决定是否丢弃，同 worldmonitor R2 strict gate）
    """
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
        logger.debug(f"丢弃未来时间戳条目: {date_str}")
        return None
    return dt


# ─── 基类 ─────────────────────────────────────────────────────────────────────

class BaseCollector:
    source_name: str = "base"

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def fetch(self, url: str, headers: dict | None = None, **kwargs) -> str:
        h = {**HEADERS, **(headers or {})}
        async with self.session.get(
            url, headers=h, timeout=aiohttp.ClientTimeout(total=15), **kwargs
        ) as resp:
            resp.raise_for_status()
            return await resp.text()

    async def collect(self) -> AsyncIterator[RawNews]:
        raise NotImplementedError


# ─── 财联社采集器（修复版） ────────────────────────────────────────────────────
#
# 旧接口 https://www.cls.cn/nodeapi/telegrams 已需要登录态或返回空数据。
# 改用电报列表接口（无需登录）：https://www.cls.cn/v1/roll/get_roll_list

class CLSCollector(BaseCollector):
    source_name = "cls"
    API_URL = "https://www.cls.cn/v1/roll/get_roll_list"

    async def collect(self) -> AsyncIterator[RawNews]:
        try:
            params = {
                "app": "CLS",
                "os": "web",
                "sv": "8.4.6",
                "rn": 20,
                "last_time": 0,
            }
            extra_headers = {
                "Referer": "https://www.cls.cn/telegraph",
                "Origin": "https://www.cls.cn",
            }
            html = await self.fetch(self.API_URL, headers=extra_headers, params=params)
            import orjson
            data = orjson.loads(html)
            items = data.get("data", {}).get("roll_data", [])
            for item in items[:20]:
                title = item.get("title", "") or item.get("brief", "")
                content = clean_text(item.get("content", title) or title)
                if not title:
                    continue
                ts = item.get("ctime", 0)
                pub = (
                    datetime.fromtimestamp(ts, tz=timezone.utc)
                    if ts
                    else datetime.now(tz=timezone.utc)
                )
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


# ─── 东方财富采集器（修复版） ─────────────────────────────────────────────────
#
# 旧接口 newsapi.eastmoney.com/kuaixun/v1/getlist_101_ajaxResult_50_1_.html
# 已返回 403 或空数据。
# 改用：https://np-listapi.eastmoney.com/comm/wap/getListInfo

class EastMoneyCollector(BaseCollector):
    source_name = "eastmoney"
    API_URL = "https://np-listapi.eastmoney.com/comm/wap/getListInfo"

    async def collect(self) -> AsyncIterator[RawNews]:
        try:
            params = {
                "client": "wap",
                "type": 1,
                "mTypeAndId": "1,317",
                "pageSize": 20,
                "pageIndex": 1,
                "callback": "",
                "_": int(datetime.now().timestamp() * 1000),
            }
            extra_headers = {
                "Referer": "https://wap.eastmoney.com/",
            }
            html = await self.fetch(self.API_URL, headers=extra_headers, params=params)
            html = html.strip()
            if html.startswith("(") and html.endswith(")"):
                html = html[1:-1]
            import orjson
            data = orjson.loads(html)
            items = (
                data.get("data", {}).get("list", [])
                or data.get("data", {}).get("LiveList", [])
                or []
            )
            for item in items[:20]:
                title = item.get("title", "")
                content = item.get("content", "") or item.get("digest", title)
                if not title:
                    continue
                pub_str = item.get("datetime", "") or item.get("showtime", "")
                pub = None
                if pub_str:
                    try:
                        pub = datetime.strptime(pub_str, "%Y-%m-%d %H:%M:%S").replace(
                            tzinfo=timezone.utc
                        )
                    except Exception:
                        pass
                pub = pub or datetime.now(tz=timezone.utc)
                yield RawNews(
                    id=make_news_id(title, self.source_name),
                    title=title,
                    content=clean_text(str(content)),
                    source=self.source_name,
                    url=item.get("url", item.get("art_url", "")),
                    published_at=pub,
                )
        except Exception as e:
            logger.warning(f"[EastMoney] 采集失败: {e}")


# ─── 新浪财经采集器（修复版） ─────────────────────────────────────────────────
#
# 旧实现依赖 CSS selector .list_009 li a，新浪财经页面结构已改变。
# 改用新浪财经直播快讯 JSON 接口，稳定且无需解析 HTML。

class SinaCollector(BaseCollector):
    source_name = "sina"
    API_URL = "https://zhibo.sina.com.cn/api/zhibo/feed"

    async def collect(self) -> AsyncIterator[RawNews]:
        try:
            params = {
                "page": 1,
                "page_size": 20,
                "zhibo_id": 152,
                "tag_id": 0,
                "dire": "f",
                "dpc": 1,
            }
            extra_headers = {
                "Referer": "https://finance.sina.com.cn/",
            }
            html = await self.fetch(self.API_URL, headers=extra_headers, params=params)
            import orjson
            data = orjson.loads(html)
            items = data.get("result", {}).get("data", {}).get("feed", {}).get("list", [])
            for item in items[:20]:
                rich_text = item.get("rich_text", "") or item.get("text", "")
                content = clean_text(rich_text)
                title = content[:80] if content else ""
                if not title:
                    continue
                create_time = item.get("create_time", "")
                pub = parse_pubdate(create_time) or datetime.now(tz=timezone.utc)
                yield RawNews(
                    id=make_news_id(title, self.source_name),
                    title=title,
                    content=content,
                    source=self.source_name,
                    url="https://finance.sina.com.cn/",
                    published_at=pub,
                )
        except Exception as e:
            logger.warning(f"[Sina] 采集失败: {e}")


# ─── RSS 聚合采集器（修复版） ─────────────────────────────────────────────────
#
# 修复说明：
# - 移除已失效的路透中文、彭博中文 RSS
# - 新增 A 股/行业专项 Google News RSS（同 worldmonitor gn() 模式）
# - 新增稳定可用的国内财经 RSS

def _gn(query: str) -> str:
    """构建 Google News RSS URL（同 worldmonitor gn() 辅助函数）"""
    return (
        f"https://news.google.com/rss/search"
        f"?q={urllib.parse.quote(query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
    )


RSS_FEEDS: list[tuple[str, str]] = [
    # ── 国内财经（稳定 RSS）──────────────────────────────────────────────────
    ("证券时报",   "https://www.stcn.com/feed"),
    ("中国证券网", "https://news.cnstock.com/news/sns_yw/rss.xml"),
    # ── Google News A股专项（参考 worldmonitor gn() 模式）──────────────────
    ("A股要闻",    _gn("A股 OR 上证指数 OR 沪深 when:1d")),
    ("龙头股",     _gn("涨停板 OR 龙头股 OR 主力资金 when:1d")),
    ("上市公司公告", _gn("上市公司 公告 OR 业绩 OR 定增 when:1d")),
    ("宏观政策",   _gn("央行 OR 证监会 OR 财政部 政策 when:1d")),
    ("北向资金",   _gn("北向资金 OR 陆股通 OR 外资 A股 when:1d")),
    # ── 可靠的英文财经 RSS──────────────────────────────────────────────────
    ("CNBC Markets", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Yahoo Finance", "https://finance.yahoo.com/rss/topstories"),
    ("财新",       _gn("site:caixin.com when:1d")),
    ("第一财经",   _gn("site:yicai.com when:1d")),
]


class RSSCollector(BaseCollector):
    source_name = "rss"

    async def _parse_feed(self, name: str, url: str) -> list[RawNews]:
        results = []
        try:
            html = await self.fetch(url, headers=RSS_HEADERS)

            # ── HTML 嗅探（同 worldmonitor looksLikeRssXml）──────────────────
            if not looks_like_rss(html):
                logger.warning(f"[RSS:{name}] 响应不是有效 RSS/Atom，跳过（可能被防火墙拦截）")
                return results

            feed = feedparser.parse(html)
            for entry in feed.entries[:10]:
                title = entry.get("title", "").strip()
                content = clean_text(entry.get("summary", title) or title)
                if not title:
                    continue

                # ── 严格日期校验（同 worldmonitor R2 strict date gate）────────
                pub_parsed = entry.get("published_parsed")
                pub_str = entry.get("published", "") or entry.get("updated", "")
                pub: datetime | None = None

                if pub_parsed:
                    try:
                        pub = datetime(*pub_parsed[:6], tzinfo=timezone.utc)
                        now = datetime.now(tz=timezone.utc)
                        if pub > now + FUTURE_TOLERANCE:
                            logger.debug(f"[RSS:{name}] 丢弃未来时间戳: {title[:40]}")
                            continue
                    except Exception:
                        pub = None

                if pub is None:
                    pub = parse_pubdate(pub_str)

                # 严格模式：无法解析日期则丢弃（避免静态机构页混入）
                if pub is None:
                    logger.debug(f"[RSS:{name}] 丢弃无时间戳条目: {title[:40]}")
                    continue

                results.append(
                    RawNews(
                        id=make_news_id(title, name),
                        title=title,
                        content=content,
                        source=name,
                        url=entry.get("link", url),
                        published_at=pub,
                    )
                )
        except aiohttp.ClientResponseError as e:
            logger.warning(f"[RSS:{name}] HTTP {e.status}: {url}")
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
        self._seen: set[str] = set()

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
                try:
                    async for news in collector.collect():
                        if news.id not in self._seen:
                            self._seen.add(news.id)
                            results.append(news)
                            logger.debug(f"采集: [{news.source}] {news.title[:40]}")
                except Exception as e:
                    logger.error(f"[{collector.source_name}] 采集器异常: {e}")

        logger.info(f"本次采集完成，共 {len(results)} 条新闻")
        return results
