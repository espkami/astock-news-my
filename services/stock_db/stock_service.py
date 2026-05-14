"""
services/stock_db/stock_service.py
A股公司数据库服务
  - 从 Tushare / AKShare 拉取公司基础数据
  - 构建业务描述的向量索引（Faiss）
  - 提供按行业/关键词检索接口
"""
from __future__ import annotations
import asyncio
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from shared.config import get_settings
from shared.models import StockCompany, Industry

settings = get_settings()

# Faiss 索引持久化路径
INDEX_DIR = Path("/app/data/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ─── 向量模型 ─────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """sentence-transformers 中文向量模型（单例）"""
    _instance: Optional["EmbeddingModel"] = None
    MODEL_NAME = "shibing624/text2vec-base-chinese"

    def __init__(self):
        logger.info("加载向量模型...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        logger.info("向量模型加载完成")

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ─── A股数据获取 ──────────────────────────────────────────────────────────────

# 预置核心行业龙头数据（生产环境由 Tushare + 人工整理替换）
BUILTIN_STOCKS: list[dict] = [
    # 半导体
    {"ts_code": "688981.SH", "name": "中芯国际", "industry": "半导体",
     "sub_industry": "晶圆代工", "market_cap": 4800,
     "main_business": "中国领先的集成电路晶圆代工企业，提供14nm至180nm制程",
     "core_products": ["晶圆代工", "14nm制程", "28nm制程"],
     "patents": ["先进制程光刻", "FinFET工艺", "存储单元设计"],
     "industry_tags": ["semiconductor", "chip_manufacturing"],
     "is_leader": True},

    {"ts_code": "002415.SZ", "name": "海康威视", "industry": "电子",
     "sub_industry": "安防设备", "market_cap": 3200,
     "main_business": "全球领先的安防产品及解决方案提供商，AI芯片自研",
     "core_products": ["安防摄像头", "AI芯片", "视频分析"],
     "patents": ["图像识别", "AI视觉", "边缘计算"],
     "industry_tags": ["ai_robot", "semiconductor"],
     "is_leader": True},

    {"ts_code": "300014.SZ", "name": "亿纬锂能", "industry": "电气设备",
     "sub_industry": "锂电池", "market_cap": 1500,
     "main_business": "动力电池和储能电池研发制造，客户覆盖宝马、戴姆勒",
     "core_products": ["动力电池", "储能电池", "圆柱电池"],
     "patents": ["电池热管理", "正极材料", "BMS系统"],
     "industry_tags": ["new_energy", "battery"],
     "is_leader": True},

    {"ts_code": "300750.SZ", "name": "宁德时代", "industry": "电气设备",
     "sub_industry": "锂电池", "market_cap": 8500,
     "main_business": "全球最大动力电池制造商，市占率超35%",
     "core_products": ["动力电池", "麒麟电池", "钠离子电池"],
     "patents": ["刀片电池", "无模组技术", "固态电池"],
     "industry_tags": ["new_energy", "battery", "energy_storage"],
     "is_leader": True},

    {"ts_code": "601012.SH", "name": "隆基绿能", "industry": "电气设备",
     "sub_industry": "光伏", "market_cap": 2100,
     "main_business": "全球最大单晶硅光伏组件制造商",
     "core_products": ["单晶硅片", "HPBC电池", "光伏组件"],
     "patents": ["PERC电池", "TOPCon", "钙钛矿"],
     "industry_tags": ["new_energy", "solar"],
     "is_leader": True},

    {"ts_code": "688111.SH", "name": "金山办公", "industry": "计算机",
     "sub_industry": "办公软件", "market_cap": 900,
     "main_business": "中国领先的办公软件服务商，WPS全球月活超6亿",
     "core_products": ["WPS Office", "金山文档", "AI助手"],
     "patents": ["文档处理算法", "协同编辑", "AI生成"],
     "industry_tags": ["ai_robot", "software"],
     "is_leader": True},

    {"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车",
     "sub_industry": "新能源汽车", "market_cap": 7800,
     "main_business": "全球新能源汽车销量第一，垂直整合电池+整车",
     "core_products": ["新能源汽车", "磷酸铁锂电池", "IGBT"],
     "patents": ["刀片电池", "DM混动", "云辇底盘"],
     "industry_tags": ["automobile", "new_energy", "semiconductor"],
     "is_leader": True},

    {"ts_code": "600519.SH", "name": "贵州茅台", "industry": "食品饮料",
     "sub_industry": "白酒", "market_cap": 22000,
     "main_business": "中国高端白酒龙头，飞天茅台定价权",
     "core_products": ["飞天茅台", "茅台1935", "茅台酒"],
     "patents": ["酱香酿造工艺", "窖藏技术"],
     "industry_tags": ["consumer", "food_beverage"],
     "is_leader": True},

    {"ts_code": "601318.SH", "name": "中国平安", "industry": "非银金融",
     "sub_industry": "保险", "market_cap": 6500,
     "main_business": "中国最大综合金融集团，保险+银行+资管",
     "core_products": ["人寿保险", "财产保险", "平安银行"],
     "patents": ["保险科技", "人脸识别风控", "AI核保"],
     "industry_tags": ["finance", "ai_robot"],
     "is_leader": True},

    {"ts_code": "688036.SH", "name": "传音控股", "industry": "通信",
     "sub_industry": "手机", "market_cap": 600,
     "main_business": "非洲市场手机龙头，占非洲智能机市占率超40%",
     "core_products": ["TECNO手机", "itel手机", "Infinix"],
     "patents": ["暗肤色拍照算法", "多卡多待"],
     "industry_tags": ["consumer", "semiconductor"],
     "is_leader": False},

    {"ts_code": "600031.SH", "name": "三一重工", "industry": "机械设备",
     "sub_industry": "工程机械", "market_cap": 1800,
     "main_business": "全球工程机械前5，挖掘机、泵车、起重机",
     "core_products": ["挖掘机", "泵车", "起重机"],
     "patents": ["液压系统", "工程机械电动化", "远程控制"],
     "industry_tags": ["real_estate", "new_energy", "ai_robot"],
     "is_leader": True},

    {"ts_code": "300015.SZ", "name": "爱尔眼科", "industry": "医疗服务",
     "sub_industry": "眼科医院", "market_cap": 1200,
     "main_business": "中国最大连锁眼科医疗机构",
     "core_products": ["近视手术", "白内障手术", "眼底病诊疗"],
     "patents": ["飞秒激光手术", "人工晶体植入"],
     "industry_tags": ["biotech", "consumer"],
     "is_leader": True},

    {"ts_code": "688041.SH", "name": "海光信息", "industry": "计算机",
     "sub_industry": "处理器芯片", "market_cap": 2200,
     "main_business": "国产高性能CPU和DCU（GPU类）研发",
     "core_products": ["海光CPU", "深算DCU", "AI训练芯片"],
     "patents": ["x86指令集兼容", "高速互联", "AI推理加速"],
     "industry_tags": ["semiconductor", "ai_robot"],
     "is_leader": True},

    {"ts_code": "600809.SH", "name": "山西汾酒", "industry": "食品饮料",
     "sub_industry": "白酒", "market_cap": 2800,
     "main_business": "清香型白酒龙头，国企改革标杆",
     "core_products": ["青花汾酒", "玻汾", "老白汾"],
     "patents": ["清香酿造工艺", "地缸发酵"],
     "industry_tags": ["consumer", "food_beverage"],
     "is_leader": True},

    {"ts_code": "002475.SZ", "name": "立讯精密", "industry": "电子",
     "sub_industry": "消费电子零部件", "market_cap": 2600,
     "main_business": "苹果最大代工厂之一，AirPods/iPhone组装",
     "core_products": ["消费电子连接器", "AirPods代工", "iPhone代工"],
     "patents": ["精密连接器", "无线充电", "精密制造"],
     "industry_tags": ["consumer", "semiconductor"],
     "is_leader": True},
]


def build_company_description(stock: dict) -> str:
    """将公司信息拼接为向量化文本"""
    return (
        f"{stock['name']} {stock['industry']} {stock['sub_industry']} "
        f"{stock['main_business']} "
        f"核心产品: {' '.join(stock['core_products'])} "
        f"核心专利: {' '.join(stock['patents'])}"
    )


# ─── Faiss向量索引 ────────────────────────────────────────────────────────────

class StockVectorIndex:
    """
    Faiss 向量索引，支持语义相似度检索
    """
    INDEX_FILE = INDEX_DIR / "stock.index"
    META_FILE  = INDEX_DIR / "stock_meta.pkl"

    def __init__(self):
        self.stocks: list[StockCompany] = []
        self.index = None
        self._loaded = False

    def build(self, stocks: list[StockCompany]):
        """构建向量索引"""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss未安装，跳过向量索引构建")
            self.stocks = stocks
            self._loaded = True
            return

        em = EmbeddingModel.get()
        texts = [
            f"{s.name} {s.main_business} "
            f"{' '.join(s.core_products)} {' '.join(s.patents)}"
            for s in stocks
        ]
        embeddings = em.encode(texts).astype("float32")
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)   # 内积（已归一化=余弦相似度）
        self.index.add(embeddings)
        self.stocks = stocks

        # 持久化
        faiss.write_index(self.index, str(self.INDEX_FILE))
        with open(self.META_FILE, "wb") as f:
            pickle.dump(stocks, f)

        logger.info(f"Faiss索引构建完成，共 {len(stocks)} 只股票")
        self._loaded = True

    def load(self) -> bool:
        """从磁盘加载索引"""
        if not self.INDEX_FILE.exists() or not self.META_FILE.exists():
            return False
        try:
            import faiss
            self.index = faiss.read_index(str(self.INDEX_FILE))
            with open(self.META_FILE, "rb") as f:
                self.stocks = pickle.load(f)
            self._loaded = True
            logger.info(f"Faiss索引加载完成，共 {len(self.stocks)} 只股票")
            return True
        except Exception as e:
            logger.warning(f"加载索引失败: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> list[tuple[StockCompany, float]]:
        """语义检索，返回 (stock, similarity_score) 列表"""
        if not self._loaded or not self.stocks:
            return []

        em = EmbeddingModel.get()
        qvec = em.encode_one(query).astype("float32").reshape(1, -1)

        if self.index is not None:
            scores, indices = self.index.search(qvec, min(top_k, len(self.stocks)))
            return [
                (self.stocks[int(idx)], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]
        else:
            # Faiss不可用时的简单关键词兜底
            query_lower = query.lower()
            results = []
            for stock in self.stocks:
                desc = f"{stock.name} {stock.main_business}".lower()
                score = sum(1 for w in query_lower.split() if w in desc) / 10
                if score > 0:
                    results.append((stock, min(score, 1.0)))
            return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]


# ─── 股票数据库服务 ───────────────────────────────────────────────────────────

class StockDBService:
    """A股公司数据库服务（单例）"""
    _instance: Optional["StockDBService"] = None

    def __init__(self):
        self.companies: list[StockCompany] = []
        self.vector_index = StockVectorIndex()
        self._industry_map: dict[str, list[StockCompany]] = {}

    @classmethod
    def get(cls) -> "StockDBService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, use_tushare: bool = False):
        """初始化：加载数据 + 构建索引"""
        # 尝试从 Tushare 加载
        if use_tushare and settings.tushare_token:
            try:
                self._load_from_tushare()
            except Exception as e:
                logger.warning(f"Tushare加载失败，使用内置数据: {e}")
                self._load_builtin()
        else:
            self._load_builtin()

        # 构建行业索引
        for company in self.companies:
            for tag in company.industry_tags:
                self._industry_map.setdefault(tag, []).append(company)

        # 向量索引
        if not self.vector_index.load():
            self.vector_index.build(self.companies)

        logger.info(f"股票数据库初始化完成，共 {len(self.companies)} 家公司")

    def _load_builtin(self):
        """加载内置龙头股数据"""
        for s in BUILTIN_STOCKS:
            self.companies.append(StockCompany(
                ts_code=s["ts_code"],
                name=s["name"],
                industry=s["industry"],
                sub_industry=s["sub_industry"],
                market_cap=s["market_cap"],
                main_business=s["main_business"],
                core_products=s["core_products"],
                patents=s["patents"],
                industry_tags=s["industry_tags"],
                is_leader=s["is_leader"],
            ))

    def _load_from_tushare(self):
        """从 Tushare 拉取数据（需要 token）"""
        import tushare as ts
        ts.set_token(settings.tushare_token)
        pro = ts.pro_api()

        # 获取股票基础信息
        df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
        logger.info(f"Tushare 获取到 {len(df)} 只A股")
        # TODO: 扩展为拉取主营业务、市值等字段
        # 此处简化处理，生产环境需要更完整的数据

    def get_by_industry(self, industry_tag: str) -> list[StockCompany]:
        return self._industry_map.get(industry_tag, [])

    def search_semantic(self, query: str, top_k: int = 10) -> list[tuple[StockCompany, float]]:
        return self.vector_index.search(query, top_k)

    def get_leaders(self) -> list[StockCompany]:
        return [c for c in self.companies if c.is_leader]

    def get_by_code(self, ts_code: str) -> Optional[StockCompany]:
        for c in self.companies:
            if c.ts_code == ts_code:
                return c
        return None
