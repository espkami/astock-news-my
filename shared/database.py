"""
shared/database.py  —  异步数据库层（PostgreSQL + SQLAlchemy 2.0）
"""
from __future__ import annotations
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, Text, JSON, Integer, Index
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine
)
from sqlalchemy.orm import DeclarativeBase

from shared.config import get_settings


# ─── Engine ──────────────────────────────────────────────────────────────────

def get_engine():
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )


_engine = None
_session_factory = None


def get_session_factory():
    global _engine, _session_factory
    if _session_factory is None:
        _engine = get_engine()
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─── ORM Models ──────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class NewsRecord(Base):
    """原始新闻 + 分类结果（合并存储）"""
    __tablename__ = "news_records"

    id              = Column(String(64), primary_key=True)   # SHA256
    title           = Column(Text, nullable=False)
    content         = Column(Text)
    source          = Column(String(32))
    url             = Column(Text)
    published_at    = Column(DateTime)
    collected_at    = Column(DateTime, default=datetime.utcnow)

    # 分类字段
    industries      = Column(JSON)           # list[str]
    event_type      = Column(String(32))
    sentiment       = Column(String(16))
    scope           = Column(String(16))
    confidence      = Column(Float)
    keywords        = Column(JSON)           # list[str]
    classified_by   = Column(String(8))      # bert | llm
    classified_at   = Column(DateTime)

    __table_args__ = (
        Index("ix_news_published", "published_at"),
        Index("ix_news_sentiment", "sentiment"),
        Index("ix_news_source", "source"),
    )


class StockRecord(Base):
    """A股公司数据"""
    __tablename__ = "stock_records"

    ts_code         = Column(String(16), primary_key=True)
    name            = Column(String(64), nullable=False)
    industry        = Column(String(64))
    sub_industry    = Column(String(64))
    market_cap      = Column(Float)
    main_business   = Column(Text)
    core_products   = Column(JSON)           # list[str]
    patents         = Column(JSON)           # list[str]
    industry_tags   = Column(JSON)           # list[str]
    is_leader       = Column(Boolean, default=False)
    embedding       = Column(JSON)           # list[float] — 768维
    updated_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_stock_industry", "industry"),
        Index("ix_stock_leader", "is_leader"),
    )


class MatchRecord(Base):
    """匹配结果"""
    __tablename__ = "match_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    news_id         = Column(String(64), nullable=False)
    news_title      = Column(Text)
    news_sentiment  = Column(String(16))
    news_event_type = Column(String(32))
    matched_stocks  = Column(JSON)           # list[MatchedStock dict]
    matched_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_match_news", "news_id"),
        Index("ix_match_time", "matched_at"),
    )


# ─── Init ────────────────────────────────────────────────────────────────────

async def init_db():
    """建表（首次启动时调用）"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ 数据库表初始化完成")
