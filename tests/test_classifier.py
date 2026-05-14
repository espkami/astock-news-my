"""
tests/test_classifier.py
分类器单元测试（无需真实API Key）
运行：pytest tests/ -v
"""
import pytest
from datetime import datetime, timezone

from shared.models import RawNews, Industry, EventType, Sentiment, Scope
from services.classifier.classifier import KeywordClassifier


def make_news(title: str, content: str = "") -> RawNews:
    return RawNews(
        id="test_" + title[:8],
        title=title,
        content=content or title,
        source="test",
        url="",
        published_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def clf():
    return KeywordClassifier()


# ─── 行业分类测试 ─────────────────────────────────────────────────────────────

class TestIndustryClassification:

    def test_semiconductor_title(self, clf):
        news = make_news("中芯国际宣布14nm芯片良率突破80%，晶圆产能大幅提升")
        result, conf = clf.classify(news)
        assert Industry.SEMICONDUCTOR in result.industries
        assert conf > 0.7

    def test_new_energy_title(self, clf):
        news = make_news("宁德时代麒麟电池量产，储能装机目标提升至100GWh")
        result, conf = clf.classify(news)
        assert Industry.NEW_ENERGY in result.industries

    def test_ai_robot_title(self, clf):
        news = make_news("英伟达发布最新AI大模型训练芯片，算力提升5倍")
        result, conf = clf.classify(news)
        assert Industry.AI_ROBOT in result.industries or Industry.SEMICONDUCTOR in result.industries

    def test_biotech_title(self, clf):
        news = make_news("某药企PD-1抗体临床三期成功，新药申请即将上市")
        result, conf = clf.classify(news)
        assert Industry.BIOTECH in result.industries

    def test_finance_title(self, clf):
        news = make_news("央行宣布降准0.5个百分点，释放流动性1.2万亿")
        result, conf = clf.classify(news)
        assert Industry.FINANCE in result.industries

    def test_unknown_returns_other(self, clf):
        news = make_news("今天天气不错，适合出门散步")
        result, conf = clf.classify(news)
        assert Industry.OTHER in result.industries
        assert conf <= 0.60


# ─── 情感分析测试 ─────────────────────────────────────────────────────────────

class TestSentimentAnalysis:

    def test_positive_sentiment(self, clf):
        news = make_news("比亚迪Q1销量创历史新高，净利润同比增长超50%")
        result, _ = clf.classify(news)
        assert result.sentiment == Sentiment.POSITIVE

    def test_negative_sentiment(self, clf):
        news = make_news("某锂电企业净利润下滑42%，业绩大幅低于预期亏损")
        result, _ = clf.classify(news)
        assert result.sentiment == Sentiment.NEGATIVE


# ─── 事件类型测试 ─────────────────────────────────────────────────────────────

class TestEventType:

    def test_policy_positive(self, clf):
        news = make_news("工信部发布新能源汽车补贴政策，支持力度超预期")
        result, _ = clf.classify(news)
        assert result.event_type == EventType.POLICY_POSITIVE

    def test_tech_breakthrough(self, clf):
        news = make_news("国内首款量子芯片研发成功，全球首次实现室温运行")
        result, _ = clf.classify(news)
        assert result.event_type == EventType.TECH_BREAKTHROUGH

    def test_merger(self, clf):
        news = make_news("华为宣布收购某激光雷达公司，涉及金额120亿元")
        result, _ = clf.classify(news)
        assert result.event_type == EventType.MERGER

    def test_geopolitics(self, clf):
        news = make_news("美国宣布对华半导体出口制裁，禁运光刻机设备")
        result, _ = clf.classify(news)
        assert result.event_type == EventType.GEOPOLITICS


# ─── 范围测试 ─────────────────────────────────────────────────────────────────

class TestScope:

    def test_global_scope(self, clf):
        news = make_news("美联储宣布维持利率不变，全球股市普遍下跌")
        result, _ = clf.classify(news)
        assert result.scope == Scope.GLOBAL

    def test_china_scope(self, clf):
        news = make_news("国家发改委发布新能源汽车产业规划，补贴延续至2026年")
        result, _ = clf.classify(news)
        assert result.scope == Scope.CHINA
