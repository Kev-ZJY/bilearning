"""SSE 进度事件流。

浏览器订阅 ``/api/tasks/{task_id}/events`` 后：
- 立即收到任务当前状态（避免「先订阅后漏掉已发生更新」的竞态）；
- 此后每次任务状态变化（submit / update）实时推送；
- status 变为 ``done`` / ``failed`` 时正常结束流；
- 空闲时定期发**注释帧心跳**（``: ping``），防止中间层掐断连接；
- 客户端断开时在 ``finally`` 里清理订阅，避免回调泄漏。

三个健壮性设计（都来自实测踩坑，改动前先读下面的说明）
--------------------------------------------------------
1. **keepalive 心跳**：只在有任务更新时才 yield 的话，对话这种
   「12 秒零更新」的阶段会被中间层的空闲超时（常见 30s）或浏览器后台
   标签页节流静默掐断。前端只看到「进度不动」，而回复其实好好在库里。
   SSE 注释帧（以 ``:`` 开头）不会触发前端的 ``onmessage``，是标准的
   保活手段。
2. **任务不存在时不静默挂死**：``get_status`` 返回 None 时（订阅早于
   submit 的竞态）如果不推任何东西，客户端会永远等在 CONNECTING 状态，
   既不报错也不推进——实测这种情况 2 秒收到 0 个事件。所以这里推一条
   ``pending`` 快照，并继续等后续 update。
3. **结构化事件透传**：``event`` 字段（step / token）随任务字典一起
   推给前端，agent 活动流和流式正文都靠它。前端按 ``data.event`` 分流，
   消费方不需要轮询别的端点。

队列里的任务字典用 ``task_id`` 字段标识（不是 ``id``），事件过滤据此匹配。
清理订阅必须用队列公开的 ``unsubscribe()``，不能直接碰 ``_subscribers``
——后者用 ``==`` 比较闭包会误删同名回调（见 queue.py 说明）。

``cancelled`` 与 ``done`` / ``failed`` 同为终态（见 ``TERMINAL_STATUSES``），
三者都要关流——区别只在**前端该做什么**：done 读回复、failed 弹错误、
cancelled 什么都不做（停止是用户自己按的，不是出错）。

事件生成器抽成模块级函数 ``event_stream``，便于脱离 HTTP 层做单元测试。
"""
from __future__ import annotations

import asyncio
import json
import queue as _q
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.queue import TERMINAL_STATUSES

router = APIRouter(tags=["events"])

# keepalive 间隔。取值权衡：
# - 太大 → 中间层（常见 30s 空闲超时）先掐断，心跳没赶上；
# - 太小 → 无谓的网络往返。
# 10 秒能稳稳抢在 30s 超时之前。
KEEPALIVE_SECONDS = 10.0

# 轮询间隔。0.25s 是「延迟感」与「空转开销」的折中。
# 注意这是**每个连接**的开销，连接数多时应改成阻塞 get + 超时。
POLL_INTERVAL = 0.25


async def event_stream(request: Request, task_id: str):
    """产出该任务的 SSE 事件流。

    把订阅回调收到的任务快照从本地缓冲队列取出来 yield，解耦回调线程与异步循环。
    """
    from app.core.queue import get_queue

    q = get_queue()
    local: _q.Queue = _q.Queue(maxsize=200)

    def on_update(task: dict) -> None:
        # 只关心本任务的状态变化。
        if task.get("task_id") == task_id:
            while True:
                try:
                    local.put_nowait(task)
                    break
                except _q.Full:
                    # Every event carries the full activity snapshot. Keep the latest,
                    # especially terminal status, so slow clients cannot hang or lose text.
                    try:
                        local.get_nowait()
                    except _q.Empty:
                        pass

    q.subscribe(on_update)

    # 立即推当前状态：订阅发生在 get_status 之前，二者之间若有 update 也会被
    # on_update 捕获，因此不会漏。先推一份快照兜底。
    initial = q.get_status(task_id)
    if initial:
        on_update(initial)
    else:
        # 任务还不存在（订阅早于 submit 的竞态）。**不能就这么死等**：
        # 不推任何东西的话客户端永远收不到第一个事件，EventSource 卡在
        # CONNECTING，既不报错也不推进——实测这种情况 2 秒收到 0 个事件。
        # 推一条「等待中」让前端进入可见的 pending 态，之后 update 补真值。
        local.put_nowait({
            "task_id": task_id,
            "status": "pending",
            "progress": 0.0,
            "message": "等待任务创建…",
            "stage": None,
            "event": "waiting",
        })

    # 上次真正 yield 数据帧的时刻，用于判断该不该发心跳。
    last_data_at = time.monotonic()
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                task = local.get_nowait()
            except _q.Empty:
                # 空闲期：到点就发心跳注释帧，防止连接被中间层掐断。
                # 注释帧对前端不可见（EventSource 不派发 onmessage），
                # 唯一作用是「让这条连接保持活着」。
                if time.monotonic() - last_data_at >= KEEPALIVE_SECONDS:
                    yield ": ping\n\n"
                    last_data_at = time.monotonic()
                await asyncio.sleep(POLL_INTERVAL)
                continue
            yield f"data: {json.dumps(task, ensure_ascii=False)}\n\n"
            last_data_at = time.monotonic()
            # 终态判定必须包含 cancelled：少了它，用户点「停止」之后
            # EventSource 会一直挂着，每 10 秒收一个心跳注释帧，
            # 直到浏览器标签页关掉才释放——而界面上的进度泡早就没了，
            # 用户无从判断连接是否还活着。
            if task.get("status") in TERMINAL_STATUSES:
                break
    finally:
        # 无论正常结束还是客户端断开，都清理订阅，防止回调在队列里无限累积。
        q.unsubscribe(on_update)


@router.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str, request: Request):
    return StreamingResponse(
        event_stream(request, task_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
