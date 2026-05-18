"""
services/collector/newsapi_collector.py
通过 newsapi.ai (EventRegistry) 采集三类新闻，每天一次。

采集范围：
  1. A股精准新闻    — conceptUri 锁定上交所/深交所
  2. 国际重要新闻   — 全球重大事件、地缘政治、国际经济
  3. 国内重要新闻   — 中国政策、经济、社会重大事件
  4. 用户自定义主题 — 前端配置的主题订阅

查询限制（免费账户）：
  - 单次 OR 关键词 ≤ 15 个
  - 每次查询消耗 1 token，2000 token/账户
  - 多 Key 轮询：额度耗尽自动切换下一个
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import List

from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from loguru import logger

from shared.models import RawNews
from shared.config import get_settings

settings = get_settings()

MAX_ITEMS_PER_QUERY = 20   # 每类新闻最多条数


# ── Key / Topic 加载 ─────────────────────────────────────────────────────────

def _load_keys() -> list[str]:
    multi = os.environ.get("NEWSAPI_AI_KEYS", "").strip()
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("NEWSAPI_AI_KEY", settings.newsapi_ai_key).strip()
    return [single] if single else []


def _load_topics() -> list[dict]:
    raw = os.environ.get("NEWSAPI_TOPICS", "").strip()
    if not raw:
        return []
    try:
        return [t for t in json.loads(raw) if t.get("query")]
    except Exception:
        return []


def _make_er(key: str) -> EventRegistry:
    return EventRegistry(apiKey=key, allowUseOfArchive=False)


# ── 三大类查询函数 ────────────────────────────────────────────────────────────

def _query_astock(er: EventRegistry, n: int) -> list[dict]:
    """A股精准：conceptUri 锁定上交所 + 深交所"""
    try:
        sse  = er.getConceptUri("Shanghai Stock Exchange")
        szse = er.getConceptUri("Shenzhen Stock Exchange")
        uris = [u for u in [sse, szse] if u]
        if not uris:
            return []
        q = QueryArticlesIter(
            conceptUri=QueryItems.OR(uris),
            lang=["zho", "eng"],
            isDuplicateFilter="skipDuplicates",
        )
        return list(q.execQuery(er, sortBy="date", maxItems=n))
    except Exception as e:
        logger.warning(f"[NewsAPI] A股查询失败: {e}")
        return []


def _query_international(er: EventRegistry, n: int) -> list[dict]:
    """国际重要新闻：地缘政治、国际经济、重大事件"""
    try:
        # 用 category 限定国际新闻，避免关键词超限
        world_uri = er.getCategoryUri("news/World")
        biz_uri   = er.getCategoryUri("business")
        q = QueryArticlesIter(
            categoryUri=QueryItems.OR([world_uri, biz_uri]),
            lang=["eng", "zho"],
            isDuplicateFilter="skipDuplicates",
        )
        return list(q.execQuery(er, sortBy="date", maxItems=n))
    except Exception as e:
        logger.warning(f"[NewsAPI] 国际新闻查询失败: {e}")
        # 备用：关键词查询
        try:
            kws = ["geopolitics", "global economy", "trade war", "Fed interest rate",
                   "oil price", "US China", "NATO", "sanctions", "IMF", "World Bank"]
            q2 = QueryArticlesIter(
                keywords=QueryItems.OR(kws[:10]),
                lang=["eng"],
                isDuplicateFilter="skipDuplicates",
            )
            return list(q2.execQuery(er, sortBy="date", maxItems=n))
        except Exception as e2:
            logger.warning(f"[NewsAPI] 国际新闻备用查询失败: {e2}")
            return []


def _query_china_domestic(er: EventRegistry, n: int) -> list[dict]:
    """国内重要新闻：中国政策、经济、社会"""
    try:
        china_uri = er.getLocationUri("China")
        # 优先取中文来源 + 中国地区
        q = QueryArticlesIter(
            sourceLocationUri=china_uri,
            lang=["zho"],
            isDuplicateFilter="skipDuplicates",
        )
        results = list(q.execQuery(er, sortBy="date", maxItems=n))
        if results:
            return results
        # 备用：英文关键词
        kws = ["China policy", "PBOC", "China economy", "RMB", "CCP",
               "China GDP", "China regulation", "Beijing policy"]
        q2 = QueryArticlesIter(
            keywords=QueryItems.OR(kws[:8]),
            lang=["eng", "zho"],
            isDuplicateFilter="skipDuplicates",
        )
        return list(q2.execQuery(er, sortBy="date", maxItems=n))
    except Exception as e:
        logger.warning(f"[NewsAPI] 国内新闻查询失败: {e}")
        return []


def _query_topic(er: EventRegistry, query: str, n: int) -> list[dict]:
    """用户自定义主题查询（关键词截断至 ≤14 个）"""
    try:
        kw_list = [k.strip() for k in re.split(r"[,，\s]+", query) if k.strip()][:14]
        if len(kw_list) == 1:
            q = QueryArticlesIter(keywords=kw_list[0], lang=["zho","eng"],
                                  isDuplicateFilter="skipDuplicates")
        else:
            q = QueryArticlesIter(keywords=QueryItems.OR(kw_list), lang=["zho","eng"],
                                  isDuplicateFilter="skipDuplicates")
        return list(q.execQuery(er, sortBy="date", maxItems=n))
    except Exception as e:
        logger.warning(f"[NewsAPI] 主题查询失败({query[:20]}): {e}")
        return []


# ── 主采集器 ─────────────────────────────────────────────────────────────────

class NewsAPICollector:
    """每天一次，采集 A股 + 国际 + 国内 + 自定义主题"""

    def collect(self, max_items: int = MAX_ITEMS_PER_QUERY) -> list[RawNews]:
        keys = _load_keys()
        if not keys:
            logger.warning("[NewsAPI] 未配置任何 Key，跳过采集")
            return []

        topics = _load_topics()
        all_arts: list[dict] = []

        # ── 三大固定类别 ──
        categories = [
            ("A股新闻",   _query_astock),
            ("国际新闻",  _query_international),
            ("国内新闻",  _query_china_domestic),
        ]
        for label, fn in categories:
            arts = self._try_keys(keys, fn, max_items, label)
            for a in arts:
                a["_category"] = label   # 标记来源类别
            all_arts.extend(arts)
            logger.info(f"[NewsAPI] {label} 采集 {len(arts)} 条")

        # ── 用户自定义主题 ──
        for topic in topics:
            query_str = topic.get("query", "").strip()
            label     = topic.get("label", query_str[:15])
            if not query_str:
                continue
            arts = self._try_keys(
                keys,
                lambda er, n, q=query_str: _query_topic(er, q, n),
                min(max_items, 10),
                f"主题:{label}",
            )
            for a in arts:
                a["_category"] = label
            all_arts.extend(arts)
            logger.info(f"[NewsAPI] 主题「{label}」采集 {len(arts)} 条")

        # ── 去重 + 转换 ──
        seen: set[str] = set()
        results: list[RawNews] = []
        for art in all_arts:
            uid = _art_id(art)
            if uid in seen:
                continue
            seen.add(uid)
            results.append(self._to_raw_news(art))

        logger.info(f"[NewsAPI] 全部采集完成，去重后 {len(results)} 条")
        return results

    def _try_keys(self, keys, fn, n, label) -> list[dict]:
        for idx, key in enumerate(keys):
            klabel = f"Key{idx+1}/{len(keys)}"
            try:
                er = _make_er(key)
                arts = fn(er, n)
                if arts is not None:
                    return arts
            except Exception as e:
                if any(w in str(e).lower() for w in ("quota","429","limit","exceeded","403")):
                    logger.warning(f"[NewsAPI] {klabel} 额度耗尽，切换下一个")
                    continue
                logger.error(f"[NewsAPI] {klabel} {label} 异常: {e}")
                continue
        logger.warning(f"[NewsAPI] {label} 所有 Key 均失效")
        return []

    def _to_raw_news(self, art: dict) -> RawNews:
        url      = art.get("url", "")
        title    = (art.get("title") or "").strip()
        body     = art.get("body") or ""
        source   = art.get("source", {}).get("title", "newsapi.ai") \
                   if isinstance(art.get("source"), dict) else "newsapi.ai"
        category = art.get("_category", "")
        pub_str  = art.get("dateTime", "")

        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except Exception:
            pub_dt = datetime.now(timezone.utc)

        summary = _make_summary(title, body)
        # content 开头标注类别，便于后续分类器区分
        content = f"【{category}】{summary}\n\n{body[:1500]}" if category else \
                  f"{summary}\n\n{body[:1500]}"

        return RawNews(
            id=_art_id(art),
            title=title,
            content=content,
            source=f"{source}·{category}" if category else source,
            url=url,
            published_at=pub_dt,
        )


def _art_id(art: dict) -> str:
    url = art.get("url", "")
    if url:
        return hashlib.md5(url.encode()).hexdigest()[:32]
    return hashlib.md5((art.get("title","") + art.get("dateTime","")).encode()).hexdigest()[:32]


def _make_summary(title: str, body: str) -> str:
    clean = re.sub(r"<[^>]+>", "", body or "").strip()
    clean = re.sub(r"\s+", " ", clean)
    if len(clean) >= 30:
        for sep in ("。", "！", "？", ". ", "! ", "? "):
            idx = clean.find(sep)
            if 20 <= idx <= 120:
                return clean[:idx + 1]
        return clean[:120] + ("…" if len(clean) > 120 else "")
    return title[:80]
