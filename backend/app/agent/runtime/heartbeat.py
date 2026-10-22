"""阻塞式 LLM 调用期间的心跳包装（视频流水线与 agent 工具循环共用）。

为什么这个模块存在
------------------
``LLMClient`` 的三个调用方法（``chat`` / ``chat_text`` / ``chat_tools``）都是
**阻塞**的：一次请求发出去，要等对端把整个响应生成完才返回。调用期间上层
拿不到任何事件，前端进度条完全静止。

这不是网络问题，是没有心跳。真实用户两次投诉「进度条完全不动」，
第一次以为是死锁。实测一次术语校验静默了 15 分钟。

所以**任何耗时 LLM 调用都必须走这个包装**，直接调 ``llm.chat()`` 就是复发。

为什么抽成模块而不是各写一份
----------------------------
视频流水线（``Pipeline._chat_with_heartbeat``）与 agent 工具循环（``ChatAgent``）
都需要心跳。复制两份必然漂移——而漂移的后果是「其中一条路径悄悄退回裸调用」，
症状又是「偶尔卡死」，极难定位。抽一份，两边共用。

心跳线程是 daemon：主调用返回或抛异常时 ``stop.set()`` 都会让它退出，不泄漏。
"""
from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any, Callable

# 心跳间隔（秒）。
#
# 权衡：前端看门狗的静默阈值是 10 分钟，30 秒有 20 倍余量，用户能清楚看到
# 「还在跑、已等待多久」。更短会把 SSE 事件刷爆（前端每次事件都 setState）。
#
# 测试要调快它时，请 patch **本模块的** HEARTBEAT_SEC（patch 使用点），
# 不要在调用方模块里另开一个别名——别名与真值会漂移，测试改了别名、
# 实际间隔没变，测试就成了恒为真的假守卫（本项目已踩过同类坑）。
HEARTBEAT_SEC = 30


def chat_with_heartbeat(emit: Callable[..., None], llm: Any,
                        messages: list[dict], *, label: str, stage: str,
                        progress: float, method: str = "chat",
                        activity_context: dict | None = None,
                        **call_kwargs) -> Any:
    """调 ``llm.<method>``，期间每 ``HEARTBEAT_SEC`` 秒发一条进度心跳。

    :param emit: 进度回调，签名 ``emit(task_id, stage, progress, message)``。
        流水线传 ``Pipeline._emit``（它还会回写 Video.status）；agent 传一个
        只更新任务队列的等价函数。
    :param llm: LLMClient 实例。
    :param method: 调哪个方法。``chat``（结构化）/ ``chat_text``（自然语言）/
        ``chat_tools``（工具轮）。心跳对三者一视同仁——阻塞性质相同。
    :param label: 心跳文案里的动作名，如「校验术语」「思考中」。
    :param stage: 心跳必须**复用调用方所在阶段**的 stage，不能自己编，
        否则进度文案会和真实阶段对不上（这正是 3.2 节那个误导性状态消息的同类问题）。
    :param progress: 同上，进度值也复用调用方的。
    """
    task_id = call_kwargs.pop("task_id", None)
    context_manager = nullcontext()
    attach_context = getattr(llm, "activity_context", None)
    if callable(attach_context) and isinstance(activity_context, dict):
        try:
            candidate_context = attach_context(**activity_context)
            # Inspect the type: Mock.__getattr__ manufactures a callable
            # activity_context method, but the returned Mock is not usable by
            # ``with``. Python resolves context-manager methods on the type, so
            # that is also the correct compatibility check for adapters.
            manager_type = type(candidate_context)
            if (callable(getattr(manager_type, "__enter__", None))
                    and callable(getattr(manager_type, "__exit__", None))):
                context_manager = candidate_context
        except Exception:
            # Observation context must never alter request or cancellation behavior.
            context_manager = nullcontext()
    stop = threading.Event()

    def beat():
        waited = HEARTBEAT_SEC
        while not stop.wait(HEARTBEAT_SEC):
            emit(task_id, stage, progress, f"{label}中（已等待 {waited} 秒）")
            waited += HEARTBEAT_SEC

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        with context_manager:
            return getattr(llm, method)(messages, **call_kwargs)
    finally:
        stop.set()
