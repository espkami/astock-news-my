# A股新闻-龙头股匹配系统

## 系统架构

```
astock-news-matcher/
├── services/
│   ├── collector/      # 新闻采集服务
│   ├── classifier/     # 新闻分类服务（两阶段：BERT + LLM）
│   ├── stock_db/       # A股公司数据库服务
│   ├── matcher/        # 智能匹配引擎
│   └── api/            # REST API 网关
├── shared/             # 公共模型/工具
├── docker/             # Docker配置
├── scripts/            # 初始化脚本
└── tests/              # 单元测试
```

## 快速启动

```bash
# 1. 复制环境变量
cp .env.example .env
# 编辑 .env，填入 API Keys

# 2. 启动所有服务
docker-compose up -d

# 3. 初始化A股数据库
docker-compose exec api python scripts/init_stock_db.py

# 4. 手动触发一次采集
curl -X POST http://localhost:8000/api/collect

# 5. 查看匹配结果
curl http://localhost:8000/api/results?limit=10
```

## 环境变量说明

| 变量 | 说明 |
|------|------|
| `ANTHROPIC_API_KEY` | Claude API Key（精分类用） |
| `TUSHARE_TOKEN` | Tushare A股数据Token |
| `REDIS_URL` | Redis连接地址 |
| `DATABASE_URL` | PostgreSQL连接地址 |
| `COLLECT_INTERVAL` | 采集间隔（秒，默认300） |
