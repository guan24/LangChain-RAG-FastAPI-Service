import os
import json
import asyncio
from langsmith import traceable
from typing import Any, List, Optional, AsyncGenerator

from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agent.agent_middleware import get_middleware
from app.agent.agent_tools import get_tools, set_current_user_id, set_thinking_callback
from app.core.logger_handler import logger
from app.services import session_manager as sm
from app.utils.prompt_loader import load_prompt

"""
Agent 构建与调用入口。

说明：
- 当前实现基于 `langchain.agents.create_agent`，返回的是可执行的 Agent Graph。
"""


class AgentFactory:
    """
    Agent 工厂类（面向 create_agent 的实现）。

    主要职责：
    - 每次调用创建全新的 Agent Graph（避免全局状态污染）
    - 动态注入工具、系统提示词、模型配置
    - 为上层提供统一的创建入口，兼容历史命名
    """

    def __init__(
        self,
        model: str = "qwen3-max",
        api_key: Optional[str] = None,
        default_tools: Optional[List[BaseTool]] = None,
        default_middleware: Optional[List] = None,
        default_system_prompt: Optional[str] = None,
    ):
        """
        初始化工厂配置（仅配置，不创建实例）
        :param model: 默认模型名称
        :param api_key: 默认 API Key（不传则从env读取）
        :param default_tools: 默认工具列表
        :param default_system_prompt: 默认系统提示词
        """
        self.model = model
        self.api_key = api_key or os.getenv("CHAT_API_KEY")
        self.default_tools = default_tools or self._get_default_tools()
        self.default_middleware = default_middleware or self._get_default_middleware()
        self.default_system_prompt = (
            default_system_prompt or self._get_default_system_prompt()
        )

    @staticmethod
    def _get_default_tools() -> List[BaseTool]:
        """获取默认工具列表"""
        return get_tools()

    def _get_default_middleware(self) -> List:
        """获取默认中间件列表"""
        return get_middleware()

    @staticmethod
    def _get_default_system_prompt() -> str:
        """获取默认系统提示词"""
        return load_prompt("main_prompt")

    def _create_chat_model(self, custom_model: Optional[str] = None):
        """内部方法：根据LLM_TYPE创建聊天模型实例"""
        llm_type = os.getenv("LLM_TYPE", "ALIYUN").upper()

        if llm_type == "OLLAMA":
            model_name = custom_model or os.getenv("OLLAMA_MODEL_NAME", self.model)
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            logger.info(f"🤖 Agent使用Ollama模型: {model_name}")

            return ChatOllama(
                model=model_name,
                base_url=base_url,
                streaming=True,
                top_p=0.7,
            )

        elif llm_type == "ALIYUN":
            api_key = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
            base_url = os.getenv("ALIYUN_BASE_URL")
            model_name = custom_model or os.getenv("ALIYUN_MODEL_NAME", self.model)

            logger.info(f"🤖 Agent使用阿里云百炼模型: {model_name}")

            return ChatTongyi(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                streaming=True,
                top_p=0.7,
            )

        else:
            raise ValueError(f"不支持的LLM_TYPE: {llm_type}，可选值: ALIYUN, OLLAMA")

    def create_agent_graph(
        self,
        custom_tools: Optional[List[BaseTool]] = None,
        custom_model: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Any:
        """
        创建全新的 Agent Graph（CompiledStateGraph）。

        :param custom_tools: 自定义工具列表（覆盖默认）
        :param custom_model: 自定义模型（覆盖默认）
        :param custom_system_prompt: 自定义系统提示词（覆盖默认）
        :param verbose: 是否打印详细日志
        :param kwargs:
            透传给 create_agent 的可选参数，未知参数会被忽略并记录警告。
        :return: create_agent 返回的可执行 Agent Graph
        """
        # 1. 创建组件（每次都重新创建，避免全局状态污染）
        chat_model = self._create_chat_model(custom_model)
        tools = custom_tools or self.default_tools
        system_prompt = custom_system_prompt or self.default_system_prompt
        middleware = self.default_middleware

        create_agent_kwargs = {}
        supported_kwargs = [
            "response_format",
            "state_schema",
            "context_schema",
            "checkpointer",
            "store",
            "interrupt_before",
            "interrupt_after",
            "name",
            "cache",
        ]
        for key in supported_kwargs:
            if key in kwargs:
                create_agent_kwargs[key] = kwargs.pop(key)

        if kwargs:
            logger.warning(
                f"create_agent 不支持这些参数，将忽略: {list(kwargs.keys())}"
            )

        # 2. 创建 Agent Graph（LangGraph CompiledStateGraph）
        return create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
            debug=verbose,
            **create_agent_kwargs,
        )

    @staticmethod
    def _stringify_content(content: Any) -> str:
        """
        将 LangChain message.content 统一转换为纯文本字符串。

        兼容三种常见输入：
        - `str`：直接返回
        - `list`：拼接多段内容（支持 `{"type": "text", "text": ...}`）
        - 其他对象：退化为 `str(...)`
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content) if content is not None else ""

    @staticmethod
    def _extract_response_from_messages(messages: List[BaseMessage]) -> str:
        """
        从消息序列中提取最终回答文本。

        规则：
        - 倒序扫描最后一个“非工具调用”的 `AIMessage`
        - 提取并返回其文本内容
        """
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                text = AgentFactory._stringify_content(message.content).strip()
                if text:
                    return text
        return ""


# 初始化全局工厂配置
agent_factory = AgentFactory()


def get_agent_graph():
    """
    获取 Agent 实例。
    """
    return agent_factory.create_agent_graph()


async def get_agent_response(
    query: str,
    history: Optional[List[tuple]] = None,
    user_id: Optional[str] = None,
    custom_tools: Optional[List[BaseTool]] = None,
    **kwargs,
    ):
    """
    获取一次完整的 Agent 响应（包含回答与工具步骤）。

    执行流程：
    1. 构建 Agent Graph
    2. 组装历史消息
    3. 以 `astream(..., stream_mode="updates")` 执行并解析中间事件
    4. 若流中未拿到最终文本，则回退到 `ainvoke` 再提取答案

    :param query: 用户查询
    :param history: 会话历史 [(user_msg, assistant_msg), ...]
    :param user_id: 用户ID
    :param custom_tools: 自定义工具（可选，用于动态切换工具）
    :param kwargs: 其他工厂参数
    :return: 响应结果
    """
    if user_id:
        set_current_user_id(user_id)

    try:
        # 1. 创建一次性 Agent Graph（每次请求独立实例）
        agent_graph = agent_factory.create_agent_graph(
            custom_tools=custom_tools, **kwargs
        )

        # 2. 构建 LangChain 消息格式的历史记录
        chat_history: List[BaseMessage] = []
        if history:
            for user_msg, assistant_msg in history:
                chat_history.append(HumanMessage(content=user_msg))
                chat_history.append(AIMessage(content=assistant_msg))

        # 3. 流式执行：
        # - model 节点产出 AIMessage（可能包含 tool_calls）
        # - tools 节点产出 ToolMessage（对应工具执行结果）
        response_text = ""
        steps = []
        pending_tool_calls = {}
        async for chunk in agent_graph.astream(
            {"messages": [*chat_history, HumanMessage(content=query)]},
            stream_mode="updates",
        ):
            if "model" in chunk:
                model_messages = chunk["model"].get("messages", [])
                for message in model_messages:
                    if not isinstance(message, AIMessage):
                        continue

                    if message.tool_calls:
                        # 先缓存一次工具调用元数据，待 ToolMessage 返回后再补全输出
                        for tool_call in message.tool_calls:
                            pending_tool_calls[tool_call.get("id")] = {
                                "thought": agent_factory._stringify_content(
                                    message.content
                                ),
                                "tool": tool_call.get("name"),
                                "tool_input": tool_call.get("args"),
                            }
                    else:
                        text = agent_factory._stringify_content(message.content).strip()
                        if text:
                            # 记录当前可用回答（通常最终回答会覆盖之前内容）
                            response_text = text

            if "tools" in chunk:
                tool_messages = chunk["tools"].get("messages", [])
                for tool_message in tool_messages:
                    if not isinstance(tool_message, ToolMessage):
                        continue

                    step = pending_tool_calls.pop(tool_message.tool_call_id, None)
                    if step is None:
                        step = {
                            "thought": "",
                            "tool": tool_message.name,
                            "tool_input": {},
                        }
                    step["tool_output"] = agent_factory._stringify_content(
                        tool_message.content
                    )

                    logger.info(
                        f"[工具调用] {step['tool']} | 入参: {step['tool_input']} | 输出: {step['tool_output']}"
                    )
                    steps.append(step)

        # 4. 兜底：如果 stream 中没有拿到最终文本，改用 ainvoke 一次性取回
        if not response_text:
            invoke_result = await agent_graph.ainvoke(
                {"messages": [*chat_history, HumanMessage(content=query)]}
            )
            response_text = agent_factory._extract_response_from_messages(
                invoke_result.get("messages", [])
            )

        return {
            "response": response_text or "抱歉，我无法理解您的请求。",
            "steps": steps,
        }

    except Exception as e:
        logger.error(f"Agent 执行错误: {str(e)}", exc_info=True)
        return {"response": f"抱歉，处理您的请求时出现了错误: {str(e)}", "steps": []}


@traceable
async def get_agent_stream_response(
    query: str,
    session_id: str,
    user_id: str,
    custom_tools: Optional[List[BaseTool]] = None,
    **kwargs,
    ) -> AsyncGenerator[str, None]:
    """
    获取 SSE 风格的流式响应（事件推送给前端）。

    说明：
    - Agent 主调用在后台任务中执行；
    - thinking_callback 将“思考事件”写入队列；
    - 生成器循环读取队列并实时 `yield data: ...`。

    :param query: 用户查询
    :param session_id: 会话 ID
    :param user_id: 用户 ID
    :param custom_tools: 自定义工具（可选）
    :param kwargs: 其他参数
    :return: 流式响应生成器
    """

    thinking_queue = asyncio.Queue() # 思考过程队列
    agent_result_holder = {"response": None, "error": None} # Agent 结果保存
    agent_done = asyncio.Event() # Agent 完成事件

    async def thinking_callback(data: dict):
        """思考过程回调函数，将事件放入队列"""
        logger.info(
            f"【思考过程】{data.get('stage', 'unknown')}: {data.get('content', '')}"
        )
        await thinking_queue.put(data)

    async def run_agent():
        """后台任务：执行 Agent 并写入最终结果。"""
        try:
            # 设置当前用户ID
            set_current_user_id(user_id)
            # 设置思考过程回调函数到上下文，
            # 使 Agent 内部调用的工具和 RAG 服务能拿到同一个 callback，实时推送事件
            set_thinking_callback(thinking_callback) 

            # 获取会话历史
            history = await sm.session_manager.get_history(session_id, user_id)
            logger.info(
                f"【Agent流式响应】获取会话历史成功，历史记录数: {len(history)}"
            )

            chat_history: List[BaseMessage] = []
            if history:
                for user_msg, assistant_msg in history:
                    chat_history.append(HumanMessage(content=user_msg))
                    chat_history.append(AIMessage(content=assistant_msg))

            # 为本次请求创建独立 Agent Graph
            agent_graph = agent_factory.create_agent_graph(
                custom_tools=custom_tools, **kwargs
            )

            # 这里采用 ainvoke，一次性拿到最终 message 序列
            invoke_result = await agent_graph.ainvoke(
                {"messages": [*chat_history, HumanMessage(content=query)]}
            )
            agent_result_holder["response"] = (
                agent_factory._extract_response_from_messages(
                    invoke_result.get("messages", [])
                )
                or "抱歉，我无法理解您的请求。"
            )
        except Exception as e:
            logger.error(f"【Agent流式响应】Agent执行失败: {e}", exc_info=True)
            agent_result_holder["error"] = str(e)
        finally:
            agent_done.set()

    # 启动 Agent 执行任务
    agent_task = asyncio.create_task(run_agent())

    try:
        logger.info(
            f"【Agent流式响应】开始处理请求，用户ID: {user_id}, 会话ID: {session_id}, 查询: {query}"
        )

        # 先发送初始响应
        yield f"data: {json.dumps({'type': 'response', 'content': '', 'session_id': session_id}, ensure_ascii=False)}\n\n"

        # 持续监听队列并实时推送思考事件，同时等待 Agent 完成
        while not agent_done.is_set():
            try:
                # 使用短超时轮询队列，实现实时推送
                event = await asyncio.wait_for(thinking_queue.get(), timeout=0.1)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                thinking_queue.task_done()
            except asyncio.TimeoutError:
                # 超时是正常的，继续等待
                continue

        # Agent 已完成，推送队列中剩余的所有思考事件
        while not thinking_queue.empty():
            try:
                event = thinking_queue.get_nowait()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                thinking_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # 等待 agent_task 完全结束
        await agent_task

        if agent_result_holder["error"]:
            error_message = f"错误: {agent_result_holder['error']}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_message, 'session_id': session_id}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            return

        response = agent_result_holder["response"]

        # 添加到会话历史
        await sm.session_manager.add_message(session_id, user_id, query, response)
        logger.info(f"【Agent流式响应】添加到会话历史成功")

        # 发送回答内容
        for char in response:
            yield f"data: {json.dumps({'type': 'response', 'content': char}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)

        # 发送结束标记
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"
        logger.info(f"【Agent流式响应】处理完成，会话ID: {session_id}")

    except Exception as e:
        logger.error(f"【Agent流式响应】处理请求失败: {e}", exc_info=True)

        # 取消 agent 任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass

        error_message = f"错误: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_message, 'session_id': session_id}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
