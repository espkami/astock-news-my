"""
shared/config.py  —  全局配置（从环境变量读取）
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = ""
    tushare_token: str = ""

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

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
