"""
tests/test_matcher.py
匹配引擎单元测试
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import numpy as np

from shared.models import (
    RawNews, ClassifiedNews,
    Industry, EventType, Sentiment, Scope,
)
from services.matcher.matcher import (
    calc_industry_score, calc_patent_score,
    calc_market_cap_weight, composite_score,
)
from services.stock_db.stock_service import StockCompany


def make_classified(
    title: str,
    industries: list[Industry],
    event_type: EventType = EventType.TECH_BREAKTHROUGH,
    sentiment: Sentiment = Sentiment.POSITIVE,
    keywords: list[str] | None = None,
) -> ClassifiedNews:
    raw = RawNews(
        id="test_match",
        title=title,
        content=title,
        source="test",
        url="",
        published_at=datetime.now(timezone.utc),
    )
    return ClassifiedNews(
        raw=raw,
        industries=industries,
        event_type=event_type,
        sentiment=sentiment,
        scope=Scope.CHINA,
        confidence=0.95,
        keywords=keywords or [],
        classified_by="test",
    )


def make_stock(
    ts_code: str = "000001.SZ",
    name: str = "测试公司",
    industry_tags: list[str] | None = None,
    patents: list[str] | None = None,
    market_cap: float = 1000.0,
    is_leader: bool = True,
) -> StockCompany:
    return StockCompany(
        ts_code=ts_code,
        name=name,
        industry="测试行业",
        sub_industry="测试子行业",
        market_cap=market_cap,
        main_business="测试主营业务",
        core_products=["产品A", "产品B"],
        patents=patents or ["专利A"],
        industry_tags=industry_tags or ["semiconductor"],
        is_leader=is_leader,
    )


# ─── 评分函数测试 ─────────────────────────────────────────────────────────────

class TestScoringFunctions:

    def test_industry_score_full_match(self):
        news = make_classified("芯片", [Industry.SEMICONDUCTOR])
        stock = make_stock(industry_tags=["semiconductor", "chip_manufacturing"])
        score = calc_industry_score(news, stock)
        assert score > 0.5

    def test_industry_score_no_match(self):
        news = make_classified("白酒", [Industry.CONSUMER])
        stock = make_stock(industry_tags=["semiconductor"])
        score = calc_industry_score(news, stock)
        assert score == 0.0

    def test_patent_score_with_keywords(self):
        news = make_classified(
            "光刻机突破", [Industry.SEMICONDUCTOR],
            keywords=["光刻", "晶圆", "制程"]
        )
        stock = make_stock(patents=["先进制程光刻", "晶圆代工"])
        score = calc_patent_score(news, stock)
        assert score > 0.0

    def test_market_cap_weight_large(self):
        stock = make_stock(market_cap=10000)   # 1万亿 → log10(10000)/5 = 0.8
        w = calc_market_cap_weight(stock)
        assert w >= 0.8

    def test_market_cap_weight_small(self):
        stock = make_stock(market_cap=50)      # 50亿
        w = calc_market_cap_weight(stock)
        assert w < 0.6

    def test_composite_score_range(self):
        score = composite_score(
            semantic=0.85,
            industry=0.70,
            patent=0.50,
            market_cap_w=0.90,
            leader_bonus=0.10,
        )
        assert 0.0 <= score <= 1.0

    def test_composite_score_zero(self):
        score = composite_score(0.0, 0.0, 0.0, 0.0, 0.0)
        assert score == 0.0


# ─── 匹配引擎集成测试（mock掉向量模型） ──────────────────────────────────────

def _make_mock_embedding():
    """返回一个不需要下载模型的 mock EmbeddingModel"""
    mock = MagicMock()
    mock.encode.return_value = np.random.rand(20, 768).astype("float32")
    mock.encode_one.return_value = np.random.rand(768).astype("float32")
    return mock


class TestMatchingEngine:

    @pytest.fixture(autouse=True)
    def setup(self):
        # Mock 向量模型，避免需要 HuggingFace 网络
        with patch("services.stock_db.stock_service.EmbeddingModel.get",
                   return_value=_make_mock_embedding()):
            from services.stock_db.stock_service import StockDBService
            # 重置单例，确保每次 setup 都重新初始化
            StockDBService._instance = None
            self.db = StockDBService.get()
            self.db.initialize(use_tushare=False)

            from services.matcher.matcher import MatchingEngine
            self.engine = MatchingEngine()
            self.engine._reason_gen = None   # 禁用LLM理由（测试无Key）
            yield

    def test_match_semiconductor_news(self):
        news = make_classified(
            "国内12英寸晶圆厂投产，半导体国产化加速",
            [Industry.SEMICONDUCTOR],
            keywords=["晶圆", "半导体", "国产化"],
        )
        result = self.engine.match(news)
        assert result.news_id == news.raw.id
        assert isinstance(result.matched_stocks, list)

    def test_match_result_sorted_by_score(self):
        news = make_classified(
            "宁德时代新型储能电池量产，成本下降30%",
            [Industry.NEW_ENERGY],
            keywords=["宁德时代", "储能", "电池"],
        )
        result = self.engine.match(news)
        if len(result.matched_stocks) > 1:
            scores = [s.score for s in result.matched_stocks]
            assert scores == sorted(scores, reverse=True)

    def test_match_returns_correct_sentiment(self):
        news = make_classified(
            "央行降准利好金融股",
            [Industry.FINANCE],
            sentiment=Sentiment.POSITIVE,
        )
        result = self.engine.match(news)
        assert result.news_sentiment == Sentiment.POSITIVE

    def test_empty_result_for_macro_news(self):
        """宏观新闻如果无行业标签映射，返回空或少量结果"""
        news = make_classified(
            "无关内容测试",
            [Industry.OTHER],
        )
        result = self.engine.match(news)
        assert result is not None   # 不应抛异常
