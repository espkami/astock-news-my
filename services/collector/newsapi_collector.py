"""
services/collector/newsapi_collector.py
通过 newsapi.ai (EventRegistry) 按主题采集新闻，提取大意。

功能：
  - 多 Key 轮询，额度耗尽自动切换
  - 按前端配置的主题列表分别搜索
  - 每条新闻生成 summary（大意），便于与 A 股个股对比
  - 无主题配置时回退到默认 A 股关键词

配置：
  NEWSAPI_AI_KEYS=key1,key2,key3   多 key 逗号分隔
  NEWSAPI_AI_KEY=key1              单 key 兼容
  NEWSAPI_TOPICS=[{"id":...,"label":...,"query":...}]  JSON 主题列表
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from loguru import logger

from shared.models import RawNews
from shared.config import get_settings

settings = get_settings()

# 默认关键词（未配置主题时使用）
DEFAULT_KEYWORDS = [
    "A股", "上证指数", "深证成指", "创业板", "科创板",
    "沪深300", "北交所", "A-share", "China stock market",
]

DEFAULT_MAX_ITEMS = 20   # 每个主题最多拉取条数（节省免费额度）


# ── Key 管理 ─────────────────────────────────────────────────────────────────

def _load_keys() -> list[str]:
    multi = os.environ.get("NEWSAPI_AI_KEYS", "").strip()
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("NEWSAPI_AI_KEY", settings.newsapi_ai_key).strip()
    return [single] if single else []


def _load_topics() -> list[dict]:
    """从环境变量读取主题列表，返回 [{"id":..,"label":..,"query":..}, ...]"""
    raw = os.environ.get("NEWSAPI_TOPICS", "").strip()
    if not raw:
        return []
    try:
        topics = json.loads(raw)
        return [t for t in topics if t.get("query")]
    except Exception:
        return []


# ── 采集器 ───────────────────────────────────────────────────────────────────

class NewsAPICollector:
    """多 Key 轮询 + 主题搜索 + 大意提取。"""

    def collect(self, max_items_per_topic: int = DEFAULT_MAX_ITEMS) -> list[RawNews]:
        keys = _load_keys()
        if not keys:
            logger.warning("[NewsAPI] 未配置任何 Key，跳过采集")
            return []

        topics = _load_topics()
        if not topics:
            # 无主题配置，退回默认 A 股关键词
            logger.info("[NewsAPI] 未配置主题，使用默认 A 股关键词")
            topics = [{"id": "default", "label": "A股默认",
                       "query": " OR ".join(DEFAULT_KEYWORDS)}]

        all_results: list[RawNews] = []
        for topic in topics:
            results = self._collect_topic(keys, topic, max_items_per_topic)
            all_results.extend(results)
            logger.info(f"[NewsAPI] 主题「{topic['label']}」采集 {len(results)} 条")

        # 按 URL 去重
        seen: set[str] = set()
        deduped = []
        for r in all_results:
            if r.url not in seen:
                seen.add(r.url)
                deduped.append(r)

        logger.info(f"[NewsAPI] 全部主题采集完成，去重后 {len(deduped)} 条")
        return deduped

    def _collect_topic(self, keys: list[str], topic: dict,
                       max_items: int) -> list[RawNews]:
        """对单个主题尝试所有 Key，返回结果列表。"""
        for idx, key in enumerate(keys):
            label = f"Key {idx+1}/{len(keys)}"
            try:
                return self._fetch(key, topic, max_items, label)
            except Exception as e:
                err = str(e).lower()
                if any(w in err for w in ("quota", "429", "limit", "exceeded", "403")):
                    logger.warning(f"[NewsAPI] {label} 额度耗尽，切换下一个")
                    continue
                logger.error(f"[NewsAPI] {label} 异常: {e}")
                continue
        logger.warning(f"[NewsAPI] 主题「{topic['label']}」所有 Key 均失效")
        return []

    def _fetch(self, key: str, topic: dict, max_items: int,
               label: str) -> list[RawNews]:
        er = EventRegistry(apiKey=key, allowUseOfArchive=False)
        q = QueryArticlesIter(
            keywords=topic["query"],
            lang=["zho", "eng"],
            isDuplicateFilter="skipDuplicates",
        )
        results: list[RawNews] = []
        for art in q.execQuery(er, sortBy="date", maxItems=max_items):
            try:
                results.append(self._to_raw_news(art, topic))
            except Exception as e:
                logger.debug(f"[NewsAPI] {label} 解析失败: {e}")
        return results

    def _to_raw_news(self, art: dict, topic: dict) -> RawNews:
        url     = art.get("url", "")
        title   = art.get("title", "").strip()
        body    = art.get("body", "")
        source  = art.get("source", {}).get("title", "newsapi.ai")
        pub_str = art.get("dateTime", "")

        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except Exception:
            pub_dt = datetime.now(timezone.utc)

        # 大意：取正文前 150 字，没有正文就用标题
        summary = _make_summary(title, body)

        # content 字段：大意 + 原文截断，便于后续分类和 A 股对比
        content = f"【{topic['label']}】{summary}\n\n{body[:1500]}"

        return RawNews(
            id=hashlib.md5(url.encode()).hexdigest()[:32],
            title=title,
            content=content,
            source=f"{source}·{topic['label']}",
            url=url,
            published_at=pub_dt,
        )


def _make_summary(title: str, body: str) -> str:
    """
    从标题 + 正文提取大意（150 字以内）。
    不调用 LLM，纯文本截取，保证速度和零成本。
    """
    # 优先用正文第一段（去除 HTML 标签）
    import re
    clean = re.sub(r"<[^>]+>", "", body).strip()
    clean = re.sub(r"\s+", " ", clean)

    if len(clean) >= 30:
        # 取第一句话，最多 120 字
        for sep in ("。", "！", "？", ". ", "! ", "? "):
            idx = clean.find(sep)
            if 20 <= idx <= 120:
                return clean[:idx + 1]
        return clean[:120] + ("…" if len(clean) > 120 else "")

    # 正文太短，用标题兜底
    return title[:80]
