"""
shared/models.py  —  跨服务共享的数据模型
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─── 枚举 ────────────────────────────────────────────────────────────────────

class Industry(str, Enum):
    SEMICONDUCTOR   = "semiconductor"
    NEW_ENERGY      = "new_energy"
    AI_ROBOT        = "ai_robot"
    BIOTECH         = "biotech"
    MILITARY        = "military"
    FINANCE         = "finance"
    CONSUMER        = "consumer"
    REAL_ESTATE     = "real_estate"
    AUTO            = "automobile"
    MATERIALS       = "new_materials"
    MACRO           = "macro"
    OTHER           = "other"

class EventType(str, Enum):
    POLICY_POSITIVE  = "policy_positive"
    POLICY_NEGATIVE  = "policy_negative"
    TECH_BREAKTHROUGH= "tech_breakthrough"
    EARNINGS_BEAT    = "earnings_beat"
    EARNINGS_MISS    = "earnings_miss"
    MERGER           = "merger_acquisition"
    MACRO_DATA       = "macro_data"
    GEOPOLITICS      = "geopolitics"
    CREDIT_RISK      = "credit_risk"
    BUSINESS_EXPAND  = "business_expansion"
    OTHER            = "other"

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"

class Scope(str, Enum):
    GLOBAL  = "global"
    CHINA   = "china"
    COMPANY = "company"


# ─── 新闻模型 ─────────────────────────────────────────────────────────────────

class RawNews(BaseModel):
    """采集服务输出的原始新闻"""
    id: str                          # SHA256(title+source)
    title: str
    content: str
    source: str                      # cls / eastmoney / sina / rss
    url: str
    published_at: datetime
    collected_at: datetime = Field(default_factory=datetime.utcnow)


class ClassifiedNews(BaseModel):
    """分类服务输出"""
    raw: RawNews
    industries: list[Industry]
    event_type: EventType
    sentiment: Sentiment
    scope: Scope
    confidence: float                # 0.0–1.0
    keywords: list[str]
    classified_by: str               # "bert" | "llm"
    classified_at: datetime = Field(default_factory=datetime.utcnow)


# ─── A股公司模型 ──────────────────────────────────────────────────────────────

class StockCompany(BaseModel):
    """A股公司基础信息"""
    ts_code: str                     # 如 000001.SZ
    name: str
    industry: str                    # 申万一级行业
    sub_industry: str                # 申万二级行业
    market_cap: float                # 亿元
    main_business: str               # 主营业务描述
    core_products: list[str]         # 核心产品/技术
    patents: list[str]               # 核心专利方向
    industry_tags: list[str]         # 我们系统的行业标签
    is_leader: bool = False          # 是否龙头（市值top3或行业地位）
    embedding: Optional[list[float]] = None  # 业务描述向量


# ─── 匹配结果 ─────────────────────────────────────────────────────────────────

class MatchedStock(BaseModel):
    """单条匹配股票"""
    ts_code: str
    name: str
    score: float                     # 综合匹配分 0.0–1.0
    semantic_score: float            # 语义向量相似度
    industry_score: float            # 行业标签匹配度
    patent_score: float              # 专利相关度
    market_cap: float
    reason: str                      # LLM生成的匹配理由


class MatchResult(BaseModel):
    """一条新闻的完整匹配结果"""
    news_id: str
    news_title: str
    news_sentiment: Sentiment
    news_event_type: EventType
    matched_stocks: list[MatchedStock]
    matched_at: datetime = Field(default_factory=datetime.utcnow)
