# 项目代码结构总结

本项目是一个基于 **FastAPI + LangChain** 构建的企业级智能对话系统，集成了先进的 **RAG（检索增强生成）** 技术。系统采用微服务架构，主要由三个核心部分组成：**后端对话服务 (Backend)**、**用户管理服务 (DjangoUserService)** 和 **前端应用 (Front)**。

---

## 1. 后端对话服务 (`backend/`)

该部分是整个系统的核心，负责处理 RAG 问答、会话管理、文档处理以及与 AI 模型的交互。

### 目录结构与功能：

| 模块 | 路径 | 作用说明 |
| :--- | :--- | :--- |
| **Agent 模块** | `app/agent/` | 智能代理核心。使用 LangChain 的 AgentExecutor 协调工具调用（如天气查询、时间获取、RAG 摘要等），支持流式输出和思考过程展示。 |
| **RAG 核心** | `app/rag/` | 检索增强生成实现。包含向量存储 (`vector_store.py`)、文档切片 (`text_spliter.py`)、混合检索器 (`retrievers/`) 以及重排序服务 (`reorder_service.py`)。 |
| **路由层** | `app/router/` | API 接口定义。处理聊天请求 (`chat.py`)、知识库管理 (`knowledge_router.py`) 和健康检查。通过 `ChatService` 编排业务逻辑。 |
| **配置管理** | `app/config/` | 存放 YAML 配置文件，包括 ChromaDB 路径、Prompt 模板路径、模型参数等。 |
| **数据模型** | `app/models/` | 定义 SQLAlchemy 模型，主要用于 MySQL 中的会话历史记录存储。 |
| **提示词管理** | `app/prompt/` | 存放各种 Prompt 文本文件（如主提示词、重排序提示词、摘要提示词）。 |
| **工具函数** | `app/utils/` | 提供通用工具，如工厂模式创建模型实例 (`factory.py`)、配置文件读取、PDF 多模态加载、图片提取等。 |
| **数据库层** | `app/db/` | 封装 Redis 和 MySQL 的连接与初始化逻辑。 |
| **缓存装饰器** | `app/cache/` | 提供基于 Redis 的异步缓存装饰器，用于优化高频查询性能。 |

### 核心技术点：
- **RAG 流程**：用户提问 -> HyDE 生成假设性回答 -> 混合检索 (Vector + BM25) -> 重排序 (Reranker) -> LLM 总结。
- **会话持久化**：利用 MySQL 存储多轮对话历史，支持长期回溯。
- **多模态支持**：支持 PDF 文档中的图片提取与关联检索。

---

## 2. 用户管理服务 (`DjangoUserService/`)

基于 Django REST Framework 构建的微服务，专门负责用户身份认证、权限管理和文件上传。

### 目录结构与功能：

| 模块 | 路径 | 作用说明 |
| :--- | :--- | :--- |
| **用户应用** | `apps/user/` | 核心用户逻辑。包含注册、登录、密码重置、Token 刷新、个人信息更新等视图 (`views.py`) 和序列化器。实现了 JWT 认证机制。 |
| **文件应用** | `apps/file/` | 处理头像等文件的上传。将文件存储在本地 Media 目录，并返回访问 URL。 |
| **工具类** | `apps/utils/` | 提供缓存工具 (`cache_utils.py`) 和限流工具 (`rate_limit_utils.py`)，增强系统安全性与性能。 |
| **认证中间件** | `apps/user/authentications.py` | 自定义 JWT 认证类，负责解析 Token、验证用户身份及黑名单管理。 |

### 核心技术点：
- **JWT 认证**：使用 Bearer Token 进行无状态身份验证，支持 Token 自动刷新。
- **跨域隔离**：不同用户的知识库在向量数据库中通过 `user_id` 进行严格隔离，确保数据安全。

---

## 3. 前端应用 (`front/`)

基于 Vue 3 + Vite 构建的现代化单页应用 (SPA)，提供友好的用户交互界面。

### 目录结构与功能：

| 模块 | 路径 | 作用说明 |
| :--- | :--- | :--- |
| **视图组件** | `src/views/` | 页面级组件。包括 AI 聊天 (`AIChat.vue`)、知识库管理 (`KnowledgeBase.vue`)、会话列表 (`Sessions.vue`)、登录/注册等。 |
| **状态管理** | `src/store/` | 使用 Pinia 管理全局状态，如用户信息 (`user.js`)、会话列表 (`session.js`) 和主题设置。 |
| **路由管理** | `src/router/` | 定义页面跳转逻辑，集成路由守卫以保护需要认证的页面。 |
| **国际化** | `src/i18n/` | 支持中英文切换，提升系统的国际化适配能力。 |
| **组合式函数** | `src/composables/` | 封装可复用的逻辑，如带认证信息的图片加载 (`useAuthImage.js`)。 |
| **API 配置** | `src/config/api.js` | 统一管理后端 API 的基础地址和接口路径。 |

### 核心技术点：
- **SSE 流式响应**：在聊天界面通过 Server-Sent Events 实时接收 AI 的思考过程和打字机效果回复。
- **Markdown 渲染**：集成 `marked` 和 `highlight.js`，完美支持代码高亮和富文本展示。

---

## 4. 数据存储与基础设施

| 组件 | 作用 |
| :--- | :--- |
| **MySQL** | 存储用户信息、会话历史记录 (Chat History)。 |
| **ChromaDB** | 轻量级向量数据库，存储文档切片及其向量表示，支持元数据过滤。 |
| **Redis** | 用于限流控制、Token 黑名单管理以及高频数据的缓存。 |
| **Ollama / DashScope** | 提供大语言模型 (LLM) 和嵌入模型 (Embedding) 的推理服务。 |

---

## 总结

该项目采用了清晰的**分层架构**：
1. **表现层 (Front)**：负责 UI 展示和用户交互。
2. **网关/路由层 (Router)**：负责请求分发、认证校验和限流。
3. **业务逻辑层 (Service/Agent/RAG)**：负责核心的 RAG 检索、Agent 决策和会话管理。
4. **数据访问层 (DB/VectorStore)**：负责与 MySQL、ChromaDB 和 Redis 的交互。

这种设计使得系统具有良好的可扩展性，可以方便地替换底层模型或增加新的业务功能。
