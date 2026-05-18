"""
shared/config.py  —  全局配置（从环境变量读取）
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = ""
    tushare_token: str = ""
    newsapi_ai_key: str = ""      # newsapi.ai 单 key（兼容旧版）
    newsapi_ai_keys: str = ""     # 多 key 逗号分隔，如 key1,key2,key3

    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/astock_news"
    redis_url: str = "redis://localhost:6379/0"

    # Service tuning
    collect_interval: int = 300
    bert_confidence_threshold: float = 0.90
    top_k_stocks: int = 5
    vector_dim: int = 768

    # Feature flags
    enable_cls: bool = True
    enable_eastmoney: bool = True
    enable_sina: bool = True
    enable_rss: bool = True
    enable_newsapi: bool = False   # newsapi.ai 默认关闭，配置 key 后手动开启

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
