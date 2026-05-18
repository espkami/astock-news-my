"""
services/collector/collector.py
新闻采集服务 — 完全基于 worldmonitor 架构重写

核心设计（来自 worldmonitor/server/worldmonitor/news/v1/）：
1. 全面改用 Google News RSS 作为主要数据源
   - worldmonitor 的 gn() 函数：用 Google News 搜索代理任意来源
   - Google News 本身不被 WAF 封锁，且聚合了所有主流财经媒体内容
   - 支持 site: 过滤、when: 时间窗口、关键词搜索
2. 直连可靠的国际 RSS（CNBC / Yahoo Finance / SEC 等）
3. 严格的 HTML 嗅探（looksLikeRssXml）
4. 严格日期门控（未来时间/无时间戳一律丢弃）
5. 重要性评分系统（来自 worldmonitor computeImportanceScore）
6. 关键词分类器（来自 worldmonitor _classifier.ts，适配 A 股）
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

# ── Chrome UA（与 worldmonitor CHROME_UA 一致）──────────────────────────────
CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

RSS_HEADERS = {
    "User-Agent": CHROME_UA,
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 未来时间宽限 1h（同 worldmonitor FUTURE_DATE_TOLERANCE_MS）
FUTURE_TOLERANCE = timedelta(hours=1)

# 新闻最大年龄 96h（同 worldmonitor 默认 NEWS_MAX_AGE_HOURS=96）
MAX_NEWS_AGE = timedelta(hours=96)

# 每个 feed 最多取条数（同 worldmonitor ITEMS_PER_FEED=5）
ITEMS_PER_FEED = 8

# ── Google News RSS 辅助函数（移植自 worldmonitor gn()）──────────────────────
def gn(query: str) -> str:
    """
    构建 Google News RSS URL。
    这是 worldmonitor 的核心技巧：用 Google News 搜索代理所有新闻源，
    绕过各站点的 WAF 封锁，同时获得 Google 聚合的权威时间戳。
    """
    return (
        "https://news.google.com/rss/search"
        f"?q={urllib.parse.quote(query)}"
        "&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
    )


def gn_en(query: str) -> str:
    """英文版 Google News RSS（用于国际财经）"""
    return (
        "https://news.google.com/rss/search"
        f"?q={urllib.parse.quote(query)}"
        "&hl=en-US&gl=US&ceid=US:en"
    )


# ── A 股新闻源定义（对标 worldmonitor VARIANT_FEEDS finance/asia 变体）────────
#
# 设计原则（来自 worldmonitor _feeds.ts 注释）：
# - 直连 RSS：用于有稳定 RSS 且不封 IP 的国际媒体（CNBC/Yahoo等）
# - Google News RSS：用于国内媒体（避免 WAF）和关键词聚合
# - site: 过滤：精确锁定权威来源（参考 worldmonitor "site:reuters.com"模式）
# - when: 时间窗口：控制新鲜度（1d/2d/3d）

ASTOCK_FEEDS: list[tuple[str, str]] = [

    # ══════════════════════════════════════════════════════════════════
    # A 股核心（Google News 聚合，绕过国内媒体 WAF）
    # ══════════════════════════════════════════════════════════════════

    # 沪深大盘 & 指数
    ("A股大盘",    gn("上证指数 OR 沪深300 OR A股 行情 when:1d")),
    # 龙头股 & 涨停
    ("龙头涨停",   gn("涨停板 OR 龙头股 OR 连板 when:1d")),
    # 主力资金
    ("主力资金",   gn("北向资金 OR 主力资金 OR 大单流入 A股 when:1d")),
    # 上市公司公告（业绩/定增/回购）
    ("公司公告",   gn("上市公司 业绩预告 OR 定增 OR 回购 OR 分红 when:1d")),
    # 行业板块热点
    ("行业热点",   gn("A股 板块 涨幅 OR 行业 轮动 when:1d")),

    # ══════════════════════════════════════════════════════════════════
    # 宏观政策（监管层动态）
    # ══════════════════════════════════════════════════════════════════

    # 证监会
    ("证监会",     gn("site:csrc.gov.cn OR 证监会 政策 监管 when:2d")),
    # 央行货币政策
    ("央行政策",   gn("央行 OR 人民银行 降准 OR 降息 OR LPR when:2d")),
    # 财政部/发改委
    ("财政政策",   gn("财政部 OR 发改委 政策 刺激 OR 补贴 when:2d")),
    # 国务院重大政策
    ("国务院",     gn("国务院 政策 OR 经济 OR 产业 when:2d")),

    # ══════════════════════════════════════════════════════════════════
    # 权威财经媒体（Google News site: 过滤，参考 worldmonitor 模式）
    # ══════════════════════════════════════════════════════════════════

    # 财联社（中国最快财经电报）
    ("财联社",     gn("site:cls.cn when:1d")),
    # 证券时报
    ("证券时报",   gn("site:stcn.com when:1d")),
    # 上海证券报
    ("上证报",     gn("site:cnstock.com when:1d")),
    # 中国证券报
    ("中证报",     gn("site:cs.com.cn when:1d")),
    # 第一财经
    ("第一财经",   gn("site:yicai.com when:1d")),
    # 财新网
    ("财新",       gn("site:caixin.com when:1d")),
    # 21世纪经济报道
    ("21财经",     gn("site:21jingji.com when:1d")),
    # 东方财富网
    ("东方财富",   gn("site:eastmoney.com when:1d")),
    # 同花顺
    ("同花顺",     gn("site:10jqka.com.cn 新闻 when:1d")),

    # ══════════════════════════════════════════════════════════════════
    # A 股专题关键词（参考 worldmonitor 关键词聚合模式）
    # ══════════════════════════════════════════════════════════════════

    # 新能源/锂电/光伏（当前热门赛道）
    ("新能源",     gn("新能源 OR 锂电 OR 光伏 A股 when:1d")),
    # 半导体/芯片
    ("半导体",     gn("半导体 OR 芯片 国产替代 A股 when:2d")),
    # 人工智能
    ("AI概念",     gn("人工智能 OR AI 大模型 A股 概念股 when:1d")),
    # 医药生物
    ("医药",       gn("医药 OR 生物医疗 OR CXO A股 when:2d")),
    # 消费/白马
    ("消费白马",   gn("消费 OR 白酒 OR 白马股 A股 when:2d")),

    # ══════════════════════════════════════════════════════════════════
    # 国际财经（直连稳定 RSS，来自 worldmonitor rss-allowed-domains.json）
    # ══════════════════════════════════════════════════════════════════

    # CNBC 市场（worldmonitor tier-2 源）
    ("CNBC Markets",   "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    # Yahoo Finance（worldmonitor 白名单）
    ("Yahoo Finance",  "https://finance.yahoo.com/rss/topstories"),
    # SEC 公告（worldmonitor gov tier-1）
    ("SEC",            "https://www.sec.gov/news/pressreleases.rss"),
    # 美联储（worldmonitor gov tier-1）
    ("Federal Reserve","https://www.federalreserve.gov/feeds/press_all.xml"),

    # ══════════════════════════════════════════════════════════════════
    # 亚太财经（影响 A 股的外部因素）
    # ══════════════════════════════════════════════════════════════════

    # 日经亚洲（Google News 代理，同 worldmonitor）
    ("Nikkei Asia",    gn_en("site:asia.nikkei.com when:2d")),
    # 南华早报（中国经济视角）
    ("SCMP China",     gn_en("site:scmp.com china economy OR markets when:2d")),
    # 中美贸易
    ("中美贸易",       gn("中美 贸易 OR 关税 OR 科技战 when:2d")),
]


# ── Source Tier（移植自 worldmonitor source-tiers.json）──────────────────────
# 用于 importanceScore 计算
SOURCE_TIERS: dict[str, int] = {
    # Tier 1：官方权威
    "证监会": 1, "央行政策": 1, "国务院": 1, "财政政策": 1,
    "SEC": 1, "Federal Reserve": 1,
    # Tier 2：主流财经媒体
    "财联社": 2, "证券时报": 2, "上证报": 2, "中证报": 2,
    "第一财经": 2, "财新": 2, "21财经": 2,
    "CNBC Markets": 2, "Yahoo Finance": 2, "Nikkei Asia": 2, "SCMP China": 2,
    # Tier 3：专题/聚合
    "东方财富": 3, "同花顺": 3,
    "A股大盘": 3, "龙头涨停": 3, "主力资金": 3, "公司公告": 3, "行业热点": 3,
    "新能源": 3, "半导体": 3, "AI概念": 3, "医药": 3, "消费白马": 3,
    "中美贸易": 3,
}


def get_source_tier(source: str) -> int:
    return SOURCE_TIERS.get(source, 4)


# ── A 股关键词分类器（移植自 worldmonitor _classifier.ts，适配 A 股）──────────
CRITICAL_KEYWORDS = {
    "熔断": "economic", "股市崩盘": "economic", "流动性危机": "economic",
    "金融危机": "economic", "银行破产": "economic", "系统性风险": "economic",
}

HIGH_KEYWORDS = {
    "大跌": "economic", "暴跌": "economic", "跌停": "economic",
    "大涨": "economic", "涨停": "economic", "暴涨": "economic",
    "降准": "economic", "降息": "economic", "加息": "economic",
    "制裁": "economic", "贸易战": "economic", "关税": "economic",
    "退市": "economic", "st摘帽": "economic", "强制退市": "economic",
    "重大重组": "economic", "借壳上市": "economic",
}

MEDIUM_KEYWORDS = {
    "业绩预增": "economic", "业绩预减": "economic", "业绩爆雷": "economic",
    "定增": "economic", "回购": "economic", "分红": "economic",
    "并购": "economic", "重组": "economic", "股权转让": "economic",
    "北向资金": "economic", "主力资金": "economic",
    "板块轮动": "economic", "热点切换": "economic",
    "政策利好": "economic", "政策利空": "economic",
    "涨幅居前": "economic", "资金流入": "economic",
}

LOW_KEYWORDS = {
    "季报": "economic", "年报": "economic", "中报": "economic",
    "股东大会": "economic", "高管变动": "economic",
    "新产品": "tech", "研发投入": "tech",
    "市值": "economic", "估值": "economic",
}

def classify_astock(title: str) -> tuple[str, str, float]:
    """
    A 股新闻关键词分类器。
    返回 (level, category, confidence)
    移植自 worldmonitor classifyByKeyword，适配 A 股场景。
    """
    lower = title.lower()

    for kw, cat in CRITICAL_KEYWORDS.items():
        if kw in lower or kw in title:
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


# ── 重要性评分（移植自 worldmonitor computeImportanceScore）──────────────────
SEVERITY_SCORES = {"critical": 100, "high": 75, "medium": 50, "low": 25, "info": 0}
SCORE_WEIGHTS = {"severity": 0.55, "source_tier": 0.20, "corroboration": 0.15, "recency": 0.10}

def compute_importance_score(
    level: str, source: str, published_at: datetime, corroboration: int = 1
) -> float:
    tier = get_source_tier(source)
    tier_score = {1: 100, 2: 75, 3: 50, 4: 25}.get(tier, 25)
    corr_score = min(corroboration, 5) * 20
    age_ms = (datetime.now(tz=timezone.utc) - published_at).total_seconds() * 1000
    recency_score = max(0.0, 1 - age_ms / (24 * 3600 * 1000)) * 100
    return round(
        SEVERITY_SCORES.get(level, 0) * SCORE_WEIGHTS["severity"]
        + tier_score * SCORE_WEIGHTS["source_tier"]
        + corr_score * SCORE_WEIGHTS["corroboration"]
        + recency_score * SCORE_WEIGHTS["recency"]
    )


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def make_news_id(title: str, source: str) -> str:
    return hashlib.sha256(f"{title}{source}".encode()).hexdigest()[:32]


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text()).strip()


def looks_like_rss(text: str) -> bool:
    """
    嗅探响应体是否为真实 RSS/Atom/RDF。
    直接移植自 worldmonitor looksLikeRssXml()。
    防止 Cloudflare 拦截页/登录墙被当作空结果缓存。
    """
    head = text[:2048].lower()
    if re.search(r"<!doctype\s+html|<html[\s>]", head):
        return False
    return bool(re.search(r"<rss[\s>]|<feed[\s>]|<rdf:rdf[\s>]", head))


def parse_pubdate(date_str: str) -> datetime | None:
    """
    解析日期字符串，严格门控（移植自 worldmonitor R2/U2）：
    - 无法解析 → None（调用方丢弃）
    - 未来时间超 1h → None
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
        return None
    return dt


# ── RSS 采集器（核心，完全对齐 worldmonitor fetchAndParseRss）────────────────

class RSSCollector:
    """
    统一 RSS 采集器。
    完全基于 worldmonitor news/v1/list-feed-digest.ts 的架构：
    - 所有数据源统一走 RSS（Google News RSS + 直连 RSS）
    - 严格 HTML 嗅探
    - 严格日期门控
    - 重要性评分
    """

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def _fetch_rss(self, url: str) -> str | None:
        """
        拉取 RSS，返回原始文本。
        HTTP 非 2xx 或响应不是 RSS → 返回 None。
        移植自 worldmonitor fetchRssText()。
        """
        try:
            async with self.session.get(
                url,
                headers=RSS_HEADERS,
                timeout=aiohttp.ClientTimeout(total=12),
                ssl=False,
            ) as resp:
                if not resp.ok:
                    logger.debug(f"[RSS] HTTP {resp.status}: {url[:60]}")
                    return None
                text = await resp.text()
                # HTML 嗅探（worldmonitor looksLikeRssXml 逻辑）
                if not looks_like_rss(text):
                    logger.warning(f"[RSS] 响应非 RSS（疑似被拦截）: {url[:60]}")
                    return None
                return text
        except Exception as e:
            logger.debug(f"[RSS] 拉取失败: {url[:60]} — {e}")
            return None

    async def parse_feed(self, name: str, url: str) -> list[RawNews]:
        """
        解析单个 RSS/Atom feed，返回 RawNews 列表。
        移植自 worldmonitor parseRssXml + buildDigest 逻辑。
        """
        results: list[RawNews] = []
        text = await self._fetch_rss(url)
        if text is None:
            return results

        feed = feedparser.parse(text)
        now = datetime.now(tz=timezone.utc)
        cutoff = now - MAX_NEWS_AGE

        parsed_count = 0
        dropped_undated = 0

        for entry in feed.entries[:ITEMS_PER_FEED]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            parsed_count += 1

            # ── 严格日期门控（worldmonitor R2/U2）────────────────────────────
            # 优先用 feedparser 解析的 published_parsed
            pub: datetime | None = None
            pub_parsed = entry.get("published_parsed")
            if pub_parsed:
                try:
                    pub = datetime(*pub_parsed[:6], tzinfo=timezone.utc)
                    if pub > now + FUTURE_TOLERANCE:
                        dropped_undated += 1
                        continue
                except Exception:
                    pub = None

            # feedparser 失败时用原始字符串
            if pub is None:
                raw_date = entry.get("published", "") or entry.get("updated", "")
                pub = parse_pubdate(raw_date)

            # 无法解析日期 → 丢弃（worldmonitor: never stamp with Date.now()）
            if pub is None:
                dropped_undated += 1
                logger.debug(f"[RSS:{name}] 丢弃无日期条目: {title[:40]}")
                continue

            # ── 新鲜度过滤（worldmonitor U3 freshness floor）─────────────────
            if pub < cutoff:
                logger.debug(f"[RSS:{name}] 丢弃过期条目({pub.date()}): {title[:40]}")
                continue

            # ── 内容提取 ──────────────────────────────────────────────────────
            summary = entry.get("summary", "") or ""
            content = clean_text(summary) if summary else title

            link = entry.get("link", url)

            # ── 关键词分类（worldmonitor classifyByKeyword 移植）─────────────
            level, category, confidence = classify_astock(title)

            # ── 重要性评分（worldmonitor computeImportanceScore 移植）──────────
            score = compute_importance_score(level, name, pub)

            results.append(RawNews(
                id=make_news_id(title, name),
                title=title,
                content=content,
                source=name,
                url=link,
                published_at=pub,
            ))

        if dropped_undated > 0:
            logger.debug(f"[RSS:{name}] 丢弃无日期条目 {dropped_undated}/{parsed_count}")

        return results

    async def collect_all(self) -> list[RawNews]:
        """
        并发采集所有 feed（移植自 worldmonitor buildDigest BATCH_CONCURRENCY 模式）。
        """
        BATCH_SIZE = 10  # 同 worldmonitor BATCH_CONCURRENCY=20，保守一些
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
    """
    统一采集入口。
    架构与 worldmonitor listFeedDigest 对齐：
    - 单一 RSSCollector 管理所有源
    - Google News RSS 作为主要数据通道，完全绕过各站点 WAF
    - 全局去重（sha256 title hash）
    """

    def __init__(self):
        self._seen: set[str] = set()

    async def collect_all(self) -> list[RawNews]:
        connector = aiohttp.TCPConnector(limit=30, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            collector = RSSCollector(session)
            raw = await collector.collect_all()

        # 去重 + 按时间倒序
        results: list[RawNews] = []
        for news in sorted(raw, key=lambda n: n.published_at, reverse=True):
            if news.id not in self._seen:
                self._seen.add(news.id)
                results.append(news)
                logger.debug(f"采集: [{news.source}] {news.title[:50]}")

        logger.info(f"本次采集完成，共 {len(results)} 条新闻（{len(ASTOCK_FEEDS)} 个 feed）")
        return results
