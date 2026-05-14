"""
services/classifier/classifier.py
两阶段新闻分类器
  阶段一：FinBERT-Chinese 本地快速分类（<10ms）
  阶段二：Claude API 精准分类（仅 confidence < threshold 时触发）
"""
from __future__ import annotations
import json
import re
from datetime import datetime
from typing import Optional

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.config import get_settings
from shared.models import (
    RawNews, ClassifiedNews,
    Industry, EventType, Sentiment, Scope,
)

settings = get_settings()


# ─── 行业关键词映射（BERT兜底用） ─────────────────────────────────────────────

INDUSTRY_KEYWORDS: dict[Industry, list[str]] = {
    Industry.SEMICONDUCTOR: [
        "芯片", "半导体", "晶圆", "光刻", "集成电路", "存储", "GPU", "CPU",
        "DRAM", "NAND", "封装", "先进制程", "台积电", "中芯", "华为海思",
    ],
    Industry.NEW_ENERGY: [
        "锂电池", "新能源", "储能", "光伏", "风电", "充电桩", "电池",
        "宁德", "比亚迪电池", "太阳能", "氢能", "燃料电池", "碳中和",
    ],
    Industry.AI_ROBOT: [
        "人工智能", "AI", "大模型", "机器人", "自动驾驶", "无人机",
        "ChatGPT", "算力", "大语言模型", "具身智能", "人形机器人",
    ],
    Industry.BIOTECH: [
        "医药", "生物", "疫苗", "临床", "新药", "抗体", "基因", "医疗器械",
        "PD-1", "CAR-T", "mRNA", "创新药", "仿制药", "医保",
    ],
    Industry.MILITARY: [
        "军工", "国防", "导弹", "战机", "舰艇", "雷达", "航天", "卫星",
        "航空发动机", "无人机军事", "歼", "运", "直升机",
    ],
    Industry.FINANCE: [
        "银行", "证券", "保险", "基金", "利率", "央行", "货币", "信贷",
        "降准", "降息", "贷款", "理财", "股市", "债券", "汇率",
    ],
    Industry.CONSUMER: [
        "消费", "零售", "电商", "白酒", "食品", "餐饮", "服装", "家电",
        "快消", "奢侈品", "双11", "618", "直播带货",
    ],
    Industry.REAL_ESTATE: [
        "房地产", "楼市", "房价", "开发商", "地产", "住房", "土地",
        "碧桂园", "万科", "恒大", "城投", "基础设施", "建筑",
    ],
    Industry.AUTO: [
        "汽车", "新能源车", "电动车", "智能驾驶", "自动驾驶",
        "比亚迪", "特斯拉", "造车新势力", "蔚来", "理想", "小鹏",
    ],
    Industry.MATERIALS: [
        "新材料", "碳纤维", "超导", "钛合金", "稀土", "锂", "钴",
        "铜", "铝", "特种钢", "化工", "石化",
    ],
}

EVENT_KEYWORDS: dict[EventType, list[str]] = {
    EventType.POLICY_POSITIVE: ["政策支持", "补贴", "扶持", "鼓励", "利好政策", "减税", "降费", "专项资金"],
    EventType.POLICY_NEGATIVE: ["监管", "处罚", "禁止", "限制", "整改", "罚款", "叫停"],
    EventType.TECH_BREAKTHROUGH: ["突破", "研发成功", "首款", "全球首", "国内首", "攻克", "发布新品", "创新"],
    EventType.EARNINGS_BEAT:    ["业绩超预期", "净利润增长", "收入大增", "超额完成"],
    EventType.EARNINGS_MISS:    ["业绩下滑", "净利润下降", "亏损", "低于预期", "业绩预警"],
    EventType.MERGER:           ["收购", "并购", "重组", "合并", "战略合作", "入股", "控股"],
    EventType.GEOPOLITICS:      ["制裁", "贸易战", "出口限制", "地缘", "俄乌", "台海", "禁运"],
    EventType.MACRO_DATA:       ["GDP", "CPI", "PPI", "PMI", "就业率", "贸易顺差", "外汇储备"],
}


# ─── 阶段一：关键词快速分类器 ────────────────────────────────────────────────

class KeywordClassifier:
    """
    基于关键词的快速分类器（模拟BERT行为）。
    生产环境可替换为真实的 FinBERT 模型推理。
    """

    def classify(self, news: RawNews) -> tuple[ClassifiedNews, float]:
        text = f"{news.title} {news.content}"

        # 行业匹配
        industry_scores: dict[Industry, int] = {}
        for industry, keywords in INDUSTRY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits:
                industry_scores[industry] = hits

        if industry_scores:
            industries = sorted(industry_scores, key=industry_scores.get, reverse=True)[:2]
            top_hits = industry_scores[industries[0]]
            confidence = min(0.60 + top_hits * 0.08, 0.95)
        else:
            industries = [Industry.OTHER]
            confidence = 0.50

        # 事件类型匹配
        event_type = EventType.OTHER
        for etype, keywords in EVENT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                event_type = etype
                break

        # 情感判断
        pos_words = ["涨", "上涨", "增长", "突破", "利好", "创新高", "超预期", "获批"]
        neg_words = ["跌", "下跌", "亏损", "利空", "风险", "违约", "制裁", "下滑"]
        pos_count = sum(1 for w in pos_words if w in text)
        neg_count = sum(1 for w in neg_words if w in text)
        if pos_count > neg_count:
            sentiment = Sentiment.POSITIVE
        elif neg_count > pos_count:
            sentiment = Sentiment.NEGATIVE
        else:
            sentiment = Sentiment.NEUTRAL

        # 范围判断
        if any(w in text for w in ["美国", "欧盟", "全球", "国际", "英伟达", "美联储"]):
            scope = Scope.GLOBAL
        elif any(w in text for w in ["公司", "企业名", "股价", "停牌"]):
            scope = Scope.COMPANY
        else:
            scope = Scope.CHINA

        import jieba
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=False)

        result = ClassifiedNews(
            raw=news,
            industries=industries,
            event_type=event_type,
            sentiment=sentiment,
            scope=scope,
            confidence=confidence,
            keywords=keywords,
            classified_by="bert",
        )
        return result, confidence


# ─── 阶段二：Claude LLM 精准分类器 ───────────────────────────────────────────

CLASSIFY_SYSTEM = """你是A股新闻分类专家。分析新闻后严格按JSON格式输出，不要任何额外文字。"""

CLASSIFY_PROMPT = """分析以下财经新闻，输出结构化分类JSON。

新闻标题：{title}
新闻内容：{content}

输出以下JSON字段（只输出JSON，无其他内容）：
{{
  "industries": ["从以下选1-2个: semiconductor/new_energy/ai_robot/biotech/military/finance/consumer/real_estate/automobile/new_materials/macro/other"],
  "event_type": "从以下选1个: policy_positive/policy_negative/tech_breakthrough/earnings_beat/earnings_miss/merger_acquisition/macro_data/geopolitics/credit_risk/business_expansion/other",
  "sentiment": "positive/negative/neutral",
  "scope": "global/china/company",
  "confidence": 0.0到1.0的浮点数,
  "keywords": ["最重要的5个关键词"],
  "reason": "20字以内的分类理由"
}}"""


class LLMClassifier:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def classify(self, news: RawNews) -> ClassifiedNews:
        content_short = news.content[:500]  # 控制token消耗
        prompt = CLASSIFY_PROMPT.format(title=news.title, content=content_short)

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = message.content[0].text.strip()

        # 提取JSON（防止模型输出多余内容）
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"LLM输出无法解析: {raw_text[:100]}")

        data = json.loads(json_match.group())

        def safe_industry(v: str) -> Industry:
            try:
                return Industry(v)
            except ValueError:
                return Industry.OTHER

        def safe_event(v: str) -> EventType:
            try:
                return EventType(v)
            except ValueError:
                return EventType.OTHER

        return ClassifiedNews(
            raw=news,
            industries=[safe_industry(i) for i in data.get("industries", ["other"])],
            event_type=safe_event(data.get("event_type", "other")),
            sentiment=Sentiment(data.get("sentiment", "neutral")),
            scope=Scope(data.get("scope", "china")),
            confidence=float(data.get("confidence", 0.9)),
            keywords=data.get("keywords", [])[:5],
            classified_by="llm",
        )


# ─── 两阶段统一调度器 ─────────────────────────────────────────────────────────

class NewsClassifier:
    """
    两阶段分类调度：
    1. 先用关键词分类器（快、便宜）
    2. 若 confidence < threshold，升级用 LLM（精准）
    """

    def __init__(self):
        self.fast = KeywordClassifier()
        self.llm: Optional[LLMClassifier] = None
        self.threshold = settings.bert_confidence_threshold

    def _get_llm(self) -> LLMClassifier:
        if self.llm is None:
            self.llm = LLMClassifier()
        return self.llm

    def classify(self, news: RawNews) -> ClassifiedNews:
        result, confidence = self.fast.classify(news)

        if confidence < self.threshold:
            logger.debug(
                f"置信度 {confidence:.2f} < {self.threshold}，升级LLM: {news.title[:30]}"
            )
            try:
                result = self._get_llm().classify(news)
                logger.debug(f"LLM分类完成: {result.industries} / {result.event_type}")
            except Exception as e:
                logger.warning(f"LLM分类失败，保留BERT结果: {e}")
        else:
            logger.debug(f"BERT分类完成 ({confidence:.2f}): {result.industries}")

        return result

    def classify_batch(self, news_list: list[RawNews]) -> list[ClassifiedNews]:
        results = []
        llm_count = 0
        for news in news_list:
            classified = self.classify(news)
            if classified.classified_by == "llm":
                llm_count += 1
            results.append(classified)

        logger.info(
            f"分类完成: {len(results)} 条, "
            f"BERT={len(results)-llm_count}, LLM={llm_count} "
            f"(LLM占比 {llm_count/max(len(results),1)*100:.1f}%)"
        )
        return results
