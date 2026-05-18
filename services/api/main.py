"""
services/api/main.py
FastAPI REST API 网关
  - 静态前端文件服务 (GET /)
  - 采集 / 分类 / 匹配流水线
  - APScheduler 定时任务
  - /api/config   动态更新运行时配置
  - /api/test-key 验证大模型 API Key
"""
from __future__ import annotations
import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import get_settings
from shared.database import get_db, init_db, NewsRecord, StockRecord, MatchRecord
from shared.models import MatchResult, ClassifiedNews
from services.collector.collector import NewsCollector
from services.classifier.classifier import NewsClassifier
from services.matcher.matcher import MatchingEngine
from services.stock_db.stock_service import StockDBService

settings   = get_settings()
STATIC_DIR = Path(__file__).parent / "static"


# ─── 全局服务实例 ─────────────────────────────────────────────────────────────

collector  = NewsCollector()
classifier = NewsClassifier()
stock_db   = StockDBService.get()
engine     = MatchingEngine()
scheduler  = AsyncIOScheduler(timezone="Asia/Shanghai")


# ─── 运行时可变配置（前端写入后立即生效，无需重启） ──────────────────────────

_runtime_cfg: dict = {}


def apply_runtime_cfg(data: dict):
    """把前端下发的配置写入环境变量 + 更新 settings 缓存"""
    global _runtime_cfg
    _runtime_cfg.update(data)

    mapping = {
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "tushare_token":     "TUSHARE_TOKEN",
        "newsapi_ai_key":    "NEWSAPI_AI_KEY",   # 单 key 兼容
        "collect_interval":  "COLLECT_INTERVAL",
        "bert_confidence_threshold": "BERT_CONFIDENCE_THRESHOLD",
        "top_k_stocks":      "TOP_K_STOCKS",
    }
    for key, env in mapping.items():
        if key in data and data[key] is not None:
            os.environ[env] = str(data[key])

    # 更新 classifier 阈值
    if "bert_confidence_threshold" in data:
        classifier.threshold = float(data["bert_confidence_threshold"])

    # 更新匹配 top-k
    if "top_k_stocks" in data:
        settings.top_k_stocks = int(data["top_k_stocks"])

    # 更新采集间隔：重新调度
    if "collect_interval" in data:
        interval = int(data["collect_interval"])
        if scheduler.running:
            scheduler.reschedule_job("news_pipeline", trigger="interval", seconds=interval)

    # 更新新闻源开关
    for src_key, attr in [("cls","enable_cls"),("eastmoney","enable_eastmoney"),
                           ("sina","enable_sina"),("rss","enable_rss"),("newsapi","enable_newsapi")]:
        if src_key in data:
            setattr(settings, attr, bool(data[src_key]))

    # 多 key 列表：合并单 key 兼容
    if "newsapi_ai_keys" in data and data["newsapi_ai_keys"]:
        keys = [k for k in data["newsapi_ai_keys"] if k and k.strip()]
        if keys:
            os.environ["NEWSAPI_AI_KEYS"] = ",".join(keys)
            os.environ["NEWSAPI_AI_KEY"]  = keys[0]   # 首个 key 兼容旧采集器
            logger.info(f"[NewsAPI] 已加载 {len(keys)} 个 Key")

    # 主题订阅：序列化为 JSON 存入环境变量，采集器读取
    if "newsapi_topics" in data and data["newsapi_topics"]:
        import json as _json
        topics_json = _json.dumps(data["newsapi_topics"], ensure_ascii=False)
        os.environ["NEWSAPI_TOPICS"] = topics_json
        logger.info(f"[NewsAPI] 已更新 {len(data['newsapi_topics'])} 个订阅主题")

    logger.info(f"运行时配置已更新: {list(data.keys())}")


# ─── 流水线主流程 ─────────────────────────────────────────────────────────────

async def run_pipeline(db: AsyncSession):
    logger.info("=== 流水线开始 ===")

    raw_news_list = await collector.collect_all()
    if not raw_news_list:
        logger.info("本次无新增新闻")
        return 0

    existing_ids: set[str] = set()
    for news in raw_news_list:
        if await db.get(NewsRecord, news.id):
            existing_ids.add(news.id)
    new_news = [n for n in raw_news_list if n.id not in existing_ids]
    logger.info(f"过滤后新增: {len(new_news)} 条")
    if not new_news:
        return 0

    classified = classifier.classify_batch(new_news)
    results    = engine.match_batch(classified)

    for cn in classified:
        db.add(NewsRecord(
            id           = cn.raw.id,
            title        = cn.raw.title,
            content      = cn.raw.content[:2000],
            source       = cn.raw.source,
            url          = cn.raw.url,
            published_at = cn.raw.published_at,
            industries   = [i.value for i in cn.industries],
            event_type   = cn.event_type.value,
            sentiment    = cn.sentiment.value,
            scope        = cn.scope.value,
            confidence   = cn.confidence,
            keywords     = cn.keywords,
            classified_by= cn.classified_by,
            classified_at= cn.classified_at,
        ))

    for mr in results:
        db.add(MatchRecord(
            news_id         = mr.news_id,
            news_title      = mr.news_title,
            news_sentiment  = mr.news_sentiment.value,
            news_event_type = mr.news_event_type.value,
            matched_stocks  = [s.model_dump() for s in mr.matched_stocks],
        ))

    await db.commit()
    logger.info(f"=== 流水线完成: {len(classified)} 分类, {len(results)} 匹配 ===")
    return len(new_news)


# ─── 生命周期 ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("服务启动中...")
    await init_db()
    stock_db.initialize(use_tushare=bool(settings.tushare_token))

    async def scheduled_pipeline():
        async for db in get_db():
            await run_pipeline(db)

    scheduler.add_job(
        scheduled_pipeline,
        trigger="interval",
        seconds=settings.collect_interval,
        id="news_pipeline",
        next_run_time=datetime.now(),
    )
    scheduler.start()
    logger.info(f"定时任务启动，间隔 {settings.collect_interval}s")
    yield
    scheduler.shutdown()
    logger.info("服务已关闭")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="A股新闻-龙头股匹配系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（JS / CSS / 图片等）
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── 前端页面 ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def frontend():
    """返回前端仪表盘 index.html"""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h2>前端文件未找到，请确认 static/index.html 存在</h2>", status_code=404)


# ─── 系统接口 ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["系统"])
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/stats", tags=["系统"], summary="系统统计")
async def get_stats(db: AsyncSession = Depends(get_db)):
    news_count  = (await db.execute(select(func.count()).select_from(NewsRecord))).scalar()
    match_count = (await db.execute(select(func.count()).select_from(MatchRecord))).scalar()
    return {
        "news_total":          news_count,
        "match_total":         match_count,
        "stock_total":         len(stock_db.companies),
        "collect_interval_sec":settings.collect_interval,
        "current_model":       _runtime_cfg.get("model", "claude-sonnet-4-20250514"),
        "provider":            _runtime_cfg.get("provider", "anthropic"),
    }


# ─── 配置接口 ─────────────────────────────────────────────────────────────────

class ConfigPayload(BaseModel):
    provider:                   Optional[str]   = None
    model:                      Optional[str]   = None
    anthropic_api_key:          Optional[str]   = None
    openai_api_key:             Optional[str]   = None
    qwen_api_key:               Optional[str]   = None
    zhipu_api_key:              Optional[str]   = None
    deepseek_api_key:           Optional[str]   = None
    custom_base_url:            Optional[str]   = None
    tushare_token:              Optional[str]   = None
    bert_confidence_threshold:  Optional[float] = None   # 0.0–1.0
    top_k_stocks:               Optional[int]   = None
    collect_interval:           Optional[int]   = None   # 秒
    classify_prompt:            Optional[str]   = None
    newsapi_ai_key:             Optional[str]   = None   # 兼容旧版单 key
    newsapi_ai_keys:            Optional[list]  = None   # 多 key 轮询
    newsapi_topics:             Optional[list]  = None   # 主题订阅列表
    cls:                        Optional[bool]  = None
    eastmoney:                  Optional[bool]  = None
    sina:                       Optional[bool]  = None
    rss:                        Optional[bool]  = None
    newsapi:                    Optional[bool]  = None


@app.post("/api/config", tags=["配置"], summary="动态更新运行时配置")
async def update_config(payload: ConfigPayload):
    """
    前端「保存配置」时调用此接口，立即生效（无需重启）。
    敏感字段（API Key）仅写入内存 + 环境变量，不持久化到数据库。
    """
    data = {k: v for k, v in payload.model_dump().items() if v is not None}

    # provider 决定哪个 key 作为 ANTHROPIC_API_KEY（兼容多厂商）
    provider = data.get("provider", _runtime_cfg.get("provider", "anthropic"))
    key_map = {
        "anthropic": "anthropic_api_key",
        "openai":    "openai_api_key",
        "qwen":      "qwen_api_key",
        "zhipu":     "zhipu_api_key",
        "deepseek":  "deepseek_api_key",
    }
    active_key_field = key_map.get(provider, "anthropic_api_key")
    if active_key_field in data:
        data["anthropic_api_key"] = data[active_key_field]   # 统一写入 classifier

    apply_runtime_cfg(data)
    return {"ok": True, "updated_keys": list(data.keys())}


@app.get("/api/config", tags=["配置"], summary="读取当前运行时配置（脱敏）")
async def read_config():
    safe = {k: v for k, v in _runtime_cfg.items()
            if "key" not in k.lower() and "token" not in k.lower()}
    safe["has_api_key"] = bool(_runtime_cfg.get("anthropic_api_key") or settings.anthropic_api_key)
    return safe


class TestKeyPayload(BaseModel):
    provider: str = "anthropic"
    api_key:  str
    base_url: Optional[str] = None


@app.post("/api/test-key", tags=["配置"], summary="测试 API Key 连通性")
async def test_api_key(payload: TestKeyPayload):
    """用最小请求验证 Key 是否可用（不计费）"""
    provider = payload.provider
    key      = payload.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="api_key 不能为空")

    try:
        if provider == "anthropic":
            import anthropic as _ant
            client = _ant.Anthropic(api_key=key)
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=8,
                messages=[{"role": "user", "content": "hi"}],
            )

        elif provider == "openai":
            import httpx
            r = httpx.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if r.status_code == 401:
                raise ValueError("Key 无效")
            r.raise_for_status()

        elif provider in ("qwen", "deepseek", "custom"):
            base = payload.base_url or (
                "https://dashscope.aliyuncs.com/compatible-mode/v1" if provider == "qwen"
                else "https://api.deepseek.com/v1"
            )
            import httpx
            r = httpx.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if r.status_code == 401:
                raise ValueError("Key 无效")

        else:
            return {"ok": True, "message": f"暂不支持自动验证 {provider}，请手动确认"}

        return {"ok": True, "message": "连接成功，Key 有效 ✓"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"连接失败: {str(e)[:120]}")


# ─── 业务接口 ─────────────────────────────────────────────────────────────────

@app.post("/api/collect", tags=["流水线"], summary="手动触发采集流水线")
async def manual_collect(db: AsyncSession = Depends(get_db)):
    try:
        collected = await run_pipeline(db)
        return {"status": "ok", "message": "流水线执行完成", "collected": collected or 0}
    except Exception as e:
        logger.error(f"手动采集失败: {e}")
        raise HTTPException(status_code=500, detail=f"采集失败: {str(e)[:200]}")


@app.get("/api/results", tags=["查询"], summary="获取匹配结果")
async def get_results(
    limit:     int           = Query(20, ge=1, le=100),
    sentiment: Optional[str] = Query(None, description="positive/negative/neutral"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(MatchRecord).order_by(desc(MatchRecord.matched_at)).limit(limit)
    if sentiment:
        stmt = stmt.where(MatchRecord.news_sentiment == sentiment)
    rows = (await db.execute(stmt)).scalars().all()
    return {
        "total": len(rows),
        "results": [
            {
                "news_id":        r.news_id,
                "news_title":     r.news_title,
                "sentiment":      r.news_sentiment,
                "event_type":     r.news_event_type,
                "matched_stocks": r.matched_stocks,
                "matched_at":     r.matched_at.isoformat() if r.matched_at else None,
            }
            for r in rows
        ],
    }


@app.get("/api/news", tags=["查询"], summary="获取新闻列表")
async def get_news(
    limit:  int           = Query(50, ge=1, le=200),
    source: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(NewsRecord).order_by(desc(NewsRecord.published_at)).limit(limit)
    if source:
        stmt = stmt.where(NewsRecord.source == source)
    rows = (await db.execute(stmt)).scalars().all()
    return {
        "total": len(rows),
        "news": [
            {
                "id":           r.id,
                "title":        r.title,
                "source":       r.source,
                "industries":   r.industries,
                "event_type":   r.event_type,
                "sentiment":    r.sentiment,
                "confidence":   r.confidence,
                "keywords":     r.keywords,
                "published_at": r.published_at.isoformat() if r.published_at else None,
            }
            for r in rows
        ],
    }


@app.get("/api/stocks", tags=["查询"], summary="获取A股公司列表")
async def get_stocks(
    leader_only: bool = False,
    industry:    Optional[str] = None,
):
    companies = stock_db.get_leaders() if leader_only else stock_db.companies
    if industry:
        companies = [c for c in companies if industry in c.industry_tags]
    return {
        "total": len(companies),
        "stocks": [
            {
                "ts_code":      c.ts_code,
                "name":         c.name,
                "industry":     c.industry,
                "market_cap":   c.market_cap,
                "is_leader":    c.is_leader,
                "industry_tags":c.industry_tags,
            }
            for c in companies
        ],
    }


@app.post("/api/match/single", tags=["流水线"], summary="对单条文本即时匹配")
async def match_single(body: dict):
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    from shared.models import RawNews
    from datetime import timezone
    raw = RawNews(
        id           = "manual_" + str(abs(hash(text)))[:8],
        title        = text[:100],
        content      = text,
        source       = "manual",
        url          = "",
        published_at = datetime.now(timezone.utc),
    )
    classified = classifier.classify(raw)
    result     = engine.match(classified)
    return {
        "classification": {
            "industries":    [i.value for i in classified.industries],
            "event_type":    classified.event_type.value,
            "sentiment":     classified.sentiment.value,
            "confidence":    classified.confidence,
            "keywords":      classified.keywords,
            "classified_by": classified.classified_by,
        },
        "matched_stocks": [s.model_dump() for s in result.matched_stocks],
    }
