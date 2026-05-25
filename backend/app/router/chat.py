from typing import List
import uuid

from fastapi.routing import APIRouter
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.agent import get_agent_stream_response
from app.router.chat_service import ChatService, get_router_service

from app.schemas.models import (
    QueryRequest,
    RAGResponse,
    RAGRequest,
    SessionResponse,
    ReorderResponse,
    ReorderRequest,
)
from app.utils.auth_utils import get_current_user_id
from app.core.success_response import success_response
from app.core.rate_limit import rate_limit

chat_router = APIRouter(prefix="/chat", tags=["chat"])


# 查询Agent流式响应
@chat_router.post("/agent/query/stream")
async def query_stream(
    request: QueryRequest,  # ·
    user_id: str = Depends(get_current_user_id),  # 依赖注入 user_id
    _: None = Depends(rate_limit(limit=10, window=60)),  # 依赖注入限流器
):
    """查询Agent流式响应"""
    session_id = request.session_id or str(uuid.uuid4())

    return StreamingResponse(
        get_agent_stream_response(
            request.query, session_id, user_id
        ),  # 获取 SSE 风格的流式响应
        media_type="text/event-stream",  # 流式响应
        headers={  # 添加响应头
            "Cache-Control": "no-cache",  # 禁用缓存
            "Connection": "keep-alive",  # 保持连接
        },
    )


# RAG检索
# response_model：声明API响应的数据结构
# FastAPI 会根据 response_model 自动生成 API 文档。
@chat_router.post("/rag/query", response_model=RAGResponse)
async def query_rag(
    request: RAGRequest,  # RAG查询请求
    user_id: str = Depends(get_current_user_id),
    router_service: ChatService = Depends(get_router_service),  # 路由服务
    _: None = Depends(rate_limit(limit=15, window=60)),
):
    """RAG检索"""
    response = await router_service.handle_rag_query(request.query, user_id)
    return success_response(data=RAGResponse(response=response))


# 获取会话信息
@chat_router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    router_service: ChatService = Depends(get_router_service),
):
    """获取会话信息，使用user_id验证"""
    history = await router_service.handle_get_session(session_id, user_id)
    return success_response(
        data=SessionResponse(session_id=session_id, history=history)
    )


# 删除会话
@chat_router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    router_service: ChatService = Depends(get_router_service),
):
    """删除会话"""
    await router_service.handle_delete_session(session_id, user_id)
    return success_response(message=f"Session {session_id} deleted successfully")


# 获取所有会话ID
@chat_router.get("/sessions")
async def get_all_sessions(router_service: ChatService = Depends(get_router_service)):
    """获取所有会话ID"""
    session_ids = await router_service.handle_get_all_sessions()
    return success_response(data={"sessions": session_ids})


# 获取用户所有会话ID
@chat_router.get("/sessions/{user_id}")
async def get_user_sessions(
    user_id: str,
    current_user_id: str = Depends(get_current_user_id),
    router_service: ChatService = Depends(get_router_service),
):
    """获取用户所有会话ID"""
    session_ids = await router_service.handle_get_user_sessions(
        user_id, current_user_id
    )
    return success_response(data={"sessions": session_ids})


# 文档重排序
@chat_router.post("/reorder", response_model=ReorderResponse)
async def reorder_documents(
    request: ReorderRequest,
    router_service: ChatService = Depends(get_router_service),
    _: None = Depends(rate_limit(limit=20, window=60)),
):
    """使用Ollama本地的嵌入模型对文档进行中文重排序"""
    sorted_docs = await router_service.handle_reorder(request.query, request.documents)
    return success_response(data=ReorderResponse(documents=sorted_docs))
