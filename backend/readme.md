# Backend 文件结构介绍文档

## 一、项目架构概述

本项目是一个基于 **FastAPI + LangChain** 的 RAG（Retrieval-Augmented Generation）智能问答服务，采用模块化设计，遵循清晰的分层架构。

### 架构层次

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                       │
│                    router/ (API 路由层)                         │
├─────────────────────────────────────────────────────────────────┤
│                        Business Layer                           │
│              agent/ (智能代理)    rag/ (检索增强生成)            │
├─────────────────────────────────────────────────────────────────┤
│                         Service Layer                           │
│              services/ (会话管理)    utils/ (工具函数)           │
├─────────────────────────────────────────────────────────────────┤
│                        Data Layer                               │
│              db/ (数据库配置)    models/ (数据模型)              │
├─────────────────────────────────────────────────────────────────┤
│                        Configuration                            │
│              config/ (配置文件)    prompt/ (提示词模板)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、文件结构详解

### 2.1 根目录文件

| 文件 | 功能说明 |
|------|----------|
| `main.py` | 应用入口，FastAPI 实例创建、路由注册、中间件配置、启动/关闭事件 |
| `pyproject.toml` | 项目配置文件，声明依赖、uv 配置、Python 版本要求 |
| `uv.lock` | 依赖锁定文件，记录精确的依赖版本 |
| `.env.example` | 环境变量模板，包含数据库、API Key 等配置示例 |
| `requirements.txt` | 依赖列表（兼容 pip） |
| `openapi.json` | OpenAPI 规范文档 |

### 2.2 app/router/ - API 路由层

负责处理 HTTP 请求，定义 API 端点，参数校验，调用业务层服务。

| 文件 | 功能说明 |
|------|----------|
| `chat.py` | 聊天相关路由：Agent 流式查询、RAG 检索、会话管理、文档重排序 |
| `knowledge_router.py` | 知识库管理路由：文档上传、删除、查询 |
| `user.py` | 用户管理路由：用户认证、信息查询 |
| `health.py` | 健康检查路由：服务状态监测 |
| `chat_service.py` | 聊天业务逻辑封装，作为路由层与核心服务的桥梁 |

**关键调用关系**：
```
chat.py ──→ chat_service.py ──→ rag_service.py / agent.py
chat.py ──→ session_manager (services/)
```

### 2.3 app/agent/ - 智能代理模块

实现基于 LangChain 的 Agent 框架，支持工具调用、流式响应、会话管理。

| 文件 | 功能说明 |
|------|----------|
| `agent.py` | Agent 工厂类，负责创建 AgentExecutor 实例，支持多种 LLM 模型（阿里云百炼/Ollama） |
| `agent_tools.py` | Agent 可用工具定义：RAG 摘要、天气查询、时间查询、用户信息、文档重排序 |
| `agent_middleware.py` | Agent 中间件，处理日志、追踪等横切关注点 |

**工具列表**：
| 工具名 | 功能 | 调用目标 |
|--------|------|----------|
| `rag_summary_tools` | RAG 文档摘要 | `RagService.rag_summary()` |
| `get_weather_tools` | 天气查询 | 外部天气 API |
| `what_time_is_now` | 获取当前时间 | 系统时间 |
| `get_user_info_tools` | 获取用户信息 | 用户服务 |
| `reorder_documents_tools` | 文档重排序 | `reorder_service` |

### 2.4 app/rag/ - 检索增强生成模块

实现 RAG 核心逻辑，包括文档检索、向量存储、文档重排序。

| 文件 | 功能说明 |
|------|----------|
| `rag_service.py` | RAG 服务主类，整合检索、重排序、摘要生成 |
| `vector_store.py` | 向量存储服务，管理 Chroma 向量数据库 |
| `retrievers/hybrid_retriever.py` | 混合检索器，结合向量检索和 BM25 |
| `retrievers/empty_retriever.py` | 空检索器，处理无文档场景 |
| `reorder_service.py` | 文档重排序服务，使用本地嵌入模型 |
| `text_spliter.py` | 文本分割器，将长文档切分为 chunks |
| `document_handler/processor.py` | 文档处理器，支持 PDF、图片等多种格式 |
| `md5_manager/md5_store.py` | MD5 管理器，用于文档去重 |
| `task_queue.py` | 任务队列，处理异步文档处理任务 |
| `sse_models.py` | SSE（Server-Sent Events）模型定义 |

**RAG 流程**：
```
用户查询 ──→ HyDE生成假设文档 ──→ 混合检索 ──→ 文档重排序 ──→ 分批总结 ──→ 最终回答
```

### 2.5 app/services/ - 服务层

提供通用服务，如会话管理、数据库连接管理。

| 文件 | 功能说明 |
|------|----------|
| `database_session_manager.py` | 基于数据库的会话管理器，存储聊天历史 |

### 2.6 app/db/ - 数据库配置

配置数据库连接，初始化数据库表结构。

| 文件 | 功能说明 |
|------|----------|
| `db_config.py` | MySQL 数据库配置，表结构定义（聊天历史表） |
| `redis_config.py` | Redis 配置，用于缓存、限流 |

### 2.7 app/cache/ - 缓存模块

提供缓存装饰器，减少重复计算。

| 文件 | 功能说明 |
|------|----------|
| `redis_decorator.py` | Redis 缓存装饰器，支持函数结果缓存 |

### 2.8 app/core/ - 核心模块

提供全局通用功能，如响应处理、日志、限流。

| 文件 | 功能说明 |
|------|----------|
| `success_response.py` | 统一成功响应封装 |
| `failed_response.py` | 统一失败响应封装 |
| `failed_response_register.py` | 异常处理器注册 |
| `rate_limit.py` | 限流中间件，基于令牌桶算法 |
| `logger_handler.py` | 日志处理器，统一日志格式 |

### 2.9 app/utils/ - 工具函数

提供通用工具函数，如配置加载、文件处理、认证等。

| 文件 | 功能说明 |
|------|----------|
| `config.py` | 配置管理，加载 YAML/环境变量 |
| `config_handler.py` | 配置处理器 |
| `factory.py` | 工厂函数，创建聊天模型实例 |
| `prompt_loader.py` | 提示词加载器，从文件读取提示词 |
| `auth_utils.py` | 认证工具，解析 JWT Token |
| `file_handler.py` | 文件处理工具 |
| `image_extractor.py` | 图片提取工具 |
| `path_tool.py` | 路径工具 |
| `pdf_multimodal_loader.py` | PDF 多模态加载器 |
| `vision_service.py` | 视觉服务，处理图片内容 |

### 2.10 app/config/ - 配置文件

YAML 格式配置文件，便于动态调整参数。

| 文件 | 功能说明 |
|------|----------|
| `agent.yaml` | Agent 配置：模型参数、工具列表 |
| `chroma.yaml` | Chroma 向量数据库配置 |
| `prompt.yaml` | 提示词配置 |
| `rag.yaml` | RAG 配置：检索参数、重排序策略 |

### 2.11 app/prompt/ - 提示词模板

存储各类提示词模板，支持动态加载。

| 文件 | 功能说明 |
|------|----------|
| `main_prompt.txt` | 主提示词，定义 Agent 角色和行为准则 |
| `rag_summarize.txt` | RAG 摘要提示词 |
| `reorder_prompt.txt` | 文档重排序提示词 |
| `report_prompt.txt` | 报告生成提示词 |

### 2.12 app/models/ - 数据模型

定义数据库数据模型。

| 文件 | 功能说明 |
|------|----------|
| `chat_history.py` | 聊天历史数据模型 |

### 2.13 app/schemas/ - API 数据结构

定义请求/响应的数据结构，使用 Pydantic 验证。

| 文件 | 功能说明 |
|------|----------|
| `models.py` | API 数据结构定义：QueryRequest、RAGResponse、SessionResponse 等 |

---

## 三、核心功能调用流程图

### 3.1 Agent 流式查询流程

```
用户请求 ──→ /chat/agent/query/stream ──→ chat.py
                                              │
                                              ▼
                                   get_agent_stream_response() ──→ agent.py
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
           session_manager            AgentFactory            thinking_callback
           (获取历史)              (创建AgentExecutor)          (实时推送思考)
                    │                         │
                    │                         ▼
                    │              agent_executor.astream()
                    │                         │
                    │                         ▼
                    │              判断是否调用工具
                    │              ┌───────────────┐
                    │              ▼               ▼
                    │         rag_summary_tools   其他工具
                    │              │
                    │              ▼
                    │         RagService.rag_summary() ──→ rag_service.py
                    │              │
                    │    ┌─────────┼─────────┐
                    │    ▼         ▼         ▼
                    │  HyDE     混合检索   文档重排序
                    │    │         │         │
                    │    └─────────┴─────────┘
                    │              │
                    └──────────────┼──────────────┘
                                   ▼
                          流式返回结果 + 保存会话
```

### 3.2 RAG 检索流程

```
用户查询
    │
    ▼
initialize_retriever() ──→ 获取动态权重(向量/BM25)
    │
    ▼
generate_hypothetical_document() ──→ HyDE生成假设文档
    │
    ▼
retrieve_document() ──→ 混合检索(向量+BM25)
    │
    ▼
reorder_documents() ──→ 使用Ollama本地模型重排序
    │
    ▼
get_documents_and_summary() ──→ 分批总结 + 最终综合
    │
    ▼
返回摘要结果
```

### 3.3 启动初始化流程

```
应用启动
    │
    ├─→ init_db() ────────────────────────→ 初始化数据库表
    │
    ├─→ init_database_session_manager() ──→ 初始化会话管理器
    │
    ├─→ connect_redis() ─────────────────→ 连接Redis缓存
    │
    └─→ check_and_download_reranker_model() → 检查重排序模型
```

---

## 四、模块依赖关系

### 依赖关系矩阵

| 模块 | 依赖模块 | 说明 |
|------|----------|------|
| `router/chat.py` | `agent/`, `rag/`, `services/`, `utils/auth_utils.py`, `core/` | 调用 Agent、RAG 服务、会话管理 |
| `agent/agent.py` | `agent/agent_tools.py`, `agent/agent_middleware.py`, `utils/prompt_loader.py`, `services/session_manager.py` | 依赖工具定义、中间件、提示词 |
| `rag/rag_service.py` | `rag/vector_store.py`, `rag/reorder_service.py`, `utils/factory.py`, `utils/prompt_loader.py` | 依赖向量存储、重排序服务、模型工厂 |
| `rag/vector_store.py` | `rag/retrievers/`, `utils/config_handler.py` | 依赖检索器、配置处理 |
| `services/database_session_manager.py` | `db/db_config.py` | 依赖数据库配置 |
| `utils/factory.py` | `utils/config.py` | 依赖配置管理 |
| `main.py` | `router/`, `db/`, `services/`, `core/`, `rag/reorder_service.py` | 整合所有模块 |

### 配置依赖链

```
.env (环境变量)
    │
    ├─→ settings (通过 dotenv 加载)
    │
    ├─→ pyproject.toml (项目配置)
    │       │
    │       └─→ uv.lock (依赖锁定)
    │
    └─→ app/config/*.yaml (业务配置)
            │
            ├─→ agent.yaml
            ├─→ chroma.yaml
            ├─→ prompt.yaml
            └─→ rag.yaml
```

---

## 五、关键技术栈

| 分类 | 技术 | 版本 |
|------|------|------|
| Web 框架 | FastAPI | 0.123.0 |
| 异步 ORM | SQLAlchemy | 2.0.48 |
| 向量数据库 | Chroma | 1.1.0 |
| LLM 框架 | LangChain | 1.2.13 |
| 模型服务 | Ollama / 阿里云百炼 | - |
| 缓存 | Redis | 7.3.0 |
| 数据库 | MySQL | - |
| 包管理 | uv | - |

---

## 六、总结

本项目采用 **分层模块化架构**，各模块职责清晰：

1. **路由层**：处理 HTTP 请求，参数校验，调用业务逻辑
2. **业务层**：实现核心业务逻辑（Agent、RAG）
3. **服务层**：提供通用服务（会话管理）
4. **数据层**：数据库配置和数据模型
5. **工具层**：通用工具函数和配置管理

核心数据流：
```
HTTP 请求 → 路由层 → 业务层（Agent/RAG）→ 数据层/外部服务 → 返回响应
```

这种设计使得代码易于维护、扩展和测试，符合现代 Web 开发最佳实践。