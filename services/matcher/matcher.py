"""
services/matcher/matcher.py
智能匹配引擎
  1. 语义向量检索（Faiss）
  2. 行业标签精确匹配
  3. 专利关键词交叉比对
  4. 综合评分排序
  5. LLM 生成匹配理由
"""
from __future__ import annotations
import re
from typing import Optional

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.config import get_settings
from shared.models import (
    ClassifiedNews, MatchedStock, MatchResult,
    Industry, Sentiment,
)
from services.stock_db.stock_service import StockDBService, StockCompany

settings = get_settings()


# ─── 行业标签映射：分类标签 → 股票行业标签 ───────────────────────────────────

INDUSTRY_TO_STOCK_TAGS: dict[Industry, list[str]] = {
    Industry.SEMICONDUCTOR: ["semiconductor", "chip_manufacturing", "ai_robot"],
    Industry.NEW_ENERGY:    ["new_energy", "battery", "energy_storage", "solar"],
    Industry.AI_ROBOT:      ["ai_robot", "software", "semiconductor"],
    Industry.BIOTECH:       ["biotech", "pharma"],
    Industry.MILITARY:      ["military", "aerospace"],
    Industry.FINANCE:       ["finance", "banking"],
    Industry.CONSUMER:      ["consumer", "food_beverage", "retail"],
    Industry.REAL_ESTATE:   ["real_estate", "construction"],
    Industry.AUTO:          ["automobile", "new_energy"],
    Industry.MATERIALS:     ["new_materials"],
    Industry.MACRO:         ["finance", "consumer", "real_estate"],
    Industry.OTHER:         [],
}


# ─── 评分函数 ─────────────────────────────────────────────────────────────────

def calc_industry_score(news: ClassifiedNews, stock: StockCompany) -> float:
    """行业标签匹配得分（精确匹配）"""
    stock_tags = set(stock.industry_tags)
    news_tags: set[str] = set()
    for ind in news.industries:
        news_tags.update(INDUSTRY_TO_STOCK_TAGS.get(ind, []))

    if not news_tags:
        return 0.0
    overlap = len(stock_tags & news_tags)
    return min(overlap / max(len(news_tags), 1), 1.0)


def calc_patent_score(news: ClassifiedNews, stock: StockCompany) -> float:
    """专利/关键词交叉比对得分"""
    if not stock.patents or not news.keywords:
        return 0.0
    patent_text = " ".join(stock.patents + stock.core_products).lower()
    kw_text = " ".join(news.keywords + [news.raw.title]).lower()
    hits = sum(1 for kw in kw_text.split() if len(kw) > 1 and kw in patent_text)
    return min(hits / 5, 1.0)


def calc_leader_bonus(stock: StockCompany) -> float:
    """龙头股溢价加分"""
    return 0.1 if stock.is_leader else 0.0


def calc_market_cap_weight(stock: StockCompany) -> float:
    """市值权重（对数归一化）"""
    import math
    if stock.market_cap <= 0:
        return 0.5
    return min(math.log10(stock.market_cap) / 5, 1.0)   # 10万亿→1.0，100亿→0.6


def composite_score(
    semantic: float,
    industry: float,
    patent: float,
    market_cap_w: float,
    leader_bonus: float,
) -> float:
    """
    综合评分公式：
    语义相似度  40%
    行业标签    30%
    专利相关度  20%
    市值权重    10%
    + 龙头加分  +10%
    """
    raw = (
        semantic    * 0.40 +
        industry    * 0.30 +
        patent      * 0.20 +
        market_cap_w* 0.10 +
        leader_bonus
    )
    return min(round(raw, 4), 1.0)


# ─── LLM 匹配理由生成 ─────────────────────────────────────────────────────────

REASON_PROMPT = """新闻：{title}
股票：{name}（{industry}）
主营：{main_business}

用15字以内说明该股票与新闻的关联理由（直接输出理由，不要编号和标点开头）："""


class ReasonGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._cache: dict[str, str] = {}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
    def generate(self, news_title: str, stock: StockCompany) -> str:
        cache_key = f"{news_title[:20]}::{stock.ts_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = REASON_PROMPT.format(
            title=news_title,
            name=stock.name,
            industry=stock.industry,
            main_business=stock.main_business[:80],
        )
        try:
            msg = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            reason = msg.content[0].text.strip()[:30]
        except Exception:
            reason = f"{stock.sub_industry}直接受益"

        self._cache[cache_key] = reason
        return reason


# ─── 匹配引擎主类 ─────────────────────────────────────────────────────────────

class MatchingEngine:
    """
    核心匹配引擎
    """

    def __init__(self):
        self.stock_db = StockDBService.get()
        self._reason_gen: Optional[ReasonGenerator] = None

    def _get_reason_gen(self) -> ReasonGenerator:
        if self._reason_gen is None:
            self._reason_gen = ReasonGenerator()
        return self._reason_gen

    def _candidate_stocks(self, news: ClassifiedNews) -> list[StockCompany]:
        """
        获取候选股票：
        1. 先按行业标签筛选
        2. 再用语义向量补充
        """
        candidates: dict[str, StockCompany] = {}

        # 行业标签筛选
        for ind in news.industries:
            tags = INDUSTRY_TO_STOCK_TAGS.get(ind, [])
            for tag in tags:
                for stock in self.stock_db.get_by_industry(tag):
                    candidates[stock.ts_code] = stock

        # 语义向量补充（取 top-20）
        query = f"{news.raw.title} {' '.join(news.keywords)}"
        for stock, _ in self.stock_db.search_semantic(query, top_k=20):
            candidates[stock.ts_code] = stock

        return list(candidates.values())

    def match(self, news: ClassifiedNews) -> MatchResult:
        """
        对一条已分类新闻进行匹配，返回 top-k 龙头股
        """
        candidates = self._candidate_stocks(news)
        if not candidates:
            logger.debug(f"无候选股票: {news.raw.title[:30]}")
            return MatchResult(
                news_id=news.raw.id,
                news_title=news.raw.title,
                news_sentiment=news.sentiment,
                news_event_type=news.event_type,
                matched_stocks=[],
            )

        # 语义相似度（向量检索已算，补全未检索部分）
        query = f"{news.raw.title} {' '.join(news.keywords)}"
        sem_map: dict[str, float] = {}
        for stock, score in self.stock_db.search_semantic(query, top_k=50):
            sem_map[stock.ts_code] = score

        # 逐个评分
        scored: list[tuple[StockCompany, dict[str, float]]] = []
        for stock in candidates:
            sem  = sem_map.get(stock.ts_code, 0.0)
            ind  = calc_industry_score(news, stock)
            pat  = calc_patent_score(news, stock)
            mktw = calc_market_cap_weight(stock)
            ldb  = calc_leader_bonus(stock)
            total = composite_score(sem, ind, pat, mktw, ldb)

            scored.append((stock, {
                "total": total,
                "semantic": sem,
                "industry": ind,
                "patent": pat,
            }))

        # 按综合分排序，取 top-k
        scored.sort(key=lambda x: x[1]["total"], reverse=True)
        top = scored[:settings.top_k_stocks]

        # 生成匹配理由（只对 score >= 0.3 的股票）
        matched_stocks = []
        use_llm_reason = bool(settings.anthropic_api_key)

        for stock, scores in top:
            if scores["total"] < 0.1:
                continue

            if use_llm_reason:
                try:
                    reason = self._get_reason_gen().generate(news.raw.title, stock)
                except Exception:
                    reason = f"{stock.sub_industry}板块受益"
            else:
                reason = f"{stock.sub_industry}板块受益"

            matched_stocks.append(MatchedStock(
                ts_code=stock.ts_code,
                name=stock.name,
                score=scores["total"],
                semantic_score=scores["semantic"],
                industry_score=scores["industry"],
                patent_score=scores["patent"],
                market_cap=stock.market_cap,
                reason=reason,
            ))

        logger.info(
            f"匹配完成: [{news.raw.title[:25]}...] → "
            f"{[s.name for s in matched_stocks]}"
        )

        return MatchResult(
            news_id=news.raw.id,
            news_title=news.raw.title,
            news_sentiment=news.sentiment,
            news_event_type=news.event_type,
            matched_stocks=matched_stocks,
        )

    def match_batch(self, news_list: list[ClassifiedNews]) -> list[MatchResult]:
        results = []
        for news in news_list:
            result = self.match(news)
            if result.matched_stocks:
                results.append(result)
        logger.info(f"批量匹配完成: {len(news_list)} 条新闻 → {len(results)} 条有效匹配")
        return results
