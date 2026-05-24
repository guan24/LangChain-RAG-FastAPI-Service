from langchain.agents import AgentState
from langchain.agents.middleware import (
    wrap_tool_call,
    wrap_model_call,
    after_model,
    before_model,
    after_agent,
    before_agent,
)
from langgraph.runtime import Runtime

from app.core.logger_handler import logger

"""
agent中间件模块
"""


@before_agent
def log_before_agent(status: AgentState, runtime: Runtime):
    """agent 运行前执行此函数"""
    logger.info(
        "[before_agent] agent启动，"
        + f"最新输入：{type(status['messages'][-1]).__name__} | {status['messages'][-1].content.strip()}，"
        + f"共{len(status['messages'])}条消息"
    )
    logger.debug(
        "[before_agent] agent启动， "
        + f"输入：{status['messages']}， "
        + f"共{len(status['messages'])}条消息"
    )


@after_agent
def log_after_agent(status: AgentState, runtime: Runtime):
    """agent 运行后执行此函数"""
    logger.info(
        "[after_agent] agent运行结束，"
        + f"最新输出：{type(status['messages'][-1]).__name__} | {status['messages'][-1].content.strip()}，"
        + f"共{len(status['messages'])}条消息"
    )
    logger.debug(
        "[after_agent] agent运行结束， "
        + f"输出：{status['messages']}， "
        + f"共{len(status['messages'])}条消息"
    )


@before_model
def log_before_model(status: AgentState, runtime: Runtime):
    """model 运行前执行此函数"""
    logger.info(
        "[before_model] model启动，"
        + f"最新输入：{type(status['messages'][-1]).__name__} | {status['messages'][-1].content.strip()}，"
        + f"共{len(status['messages'])}条消息"
    )
    logger.debug(
        "[before_model] model启动， "
        + f"输入：{status['messages']}， "
        + f"共{len(status['messages'])}条消息"
    )


@after_model
def log_after_model(status: AgentState, runtime: Runtime):
    """model 运行后执行此函数"""
    logger.info(
        "[after_model] model运行结束，"
        + f"最新输出：{type(status['messages'][-1]).__name__} | {status['messages'][-1].content.strip()}，"
        + f"共{len(status['messages'])}条消息"
    )
    logger.debug(
        "[after_model] model运行结束， "
        + f"输出：{status['messages']}， "
        + f"共{len(status['messages'])}条消息"
    )


@wrap_model_call
async def model_call_hook(request, handler):
    """model 调用前执行此函数"""
    logger.info("模型调用了")
    return await handler(request)


@wrap_tool_call
async def tool_call_hook(request, handler):
    """tool 调用前执行此函数"""
    logger.info(
        f"工具{request.tool_call['name']}调用了, 传入参数{request.tool_call['args']}"
    )
    return await handler(request)


def get_middleware():
    """返回本模块的所有中间件"""
    return [
        log_before_agent,
        log_after_agent,
        log_before_model,
        log_after_model,
        model_call_hook,
        tool_call_hook,
    ]
