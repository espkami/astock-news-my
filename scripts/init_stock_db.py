"""
scripts/init_stock_db.py
初始化脚本：建表 + 写入内置A股数据 + 构建Faiss索引
运行方式：
    docker-compose exec api python scripts/init_stock_db.py
"""
import asyncio
import sys
import os

# 保证 /app 在 path 里
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from shared.database import init_db, get_session_factory, StockRecord
from services.stock_db.stock_service import StockDBService, BUILTIN_STOCKS


async def main():
    logger.info("=== 初始化数据库 ===")

    # 1. 建表
    await init_db()

    # 2. 写入 A 股公司数据
    factory = get_session_factory()
    async with factory() as session:
        inserted = 0
        for s in BUILTIN_STOCKS:
            existing = await session.get(StockRecord, s["ts_code"])
            if existing:
                logger.debug(f"已存在，跳过: {s['name']}")
                continue
            record = StockRecord(
                ts_code     = s["ts_code"],
                name        = s["name"],
                industry    = s["industry"],
                sub_industry= s["sub_industry"],
                market_cap  = s["market_cap"],
                main_business = s["main_business"],
                core_products = s["core_products"],
                patents     = s["patents"],
                industry_tags = s["industry_tags"],
                is_leader   = s["is_leader"],
            )
            session.add(record)
            inserted += 1
        await session.commit()
        logger.info(f"写入 {inserted} 家公司（已存在的跳过）")

    # 3. 初始化向量索引
    logger.info("=== 构建 Faiss 向量索引 ===")
    svc = StockDBService.get()
    svc.initialize(use_tushare=False)

    logger.info("=== 初始化完成 ✅ ===")


if __name__ == "__main__":
    asyncio.run(main())
