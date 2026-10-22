"""任务控制端点：中止（cancel）。

为什么单独一个模块而不是塞进 events.py
--------------------------------------
``events.py`` 的职责是**读**（SSE 把任务状态推给浏览器），而这里是**写**
（外部要求任务停下来）。把取消做成「通过 SSE 推一个特殊事件」看着省一个端点，
实际上要求前端「必须保持连接才能取消」——而 SSE 恰恰是最容易断的一环
（切项目、后台标签页节流、网络抖动）。断流时用户按停止将毫无反应，
这正是最需要它可靠的场景。所以取消必须走一个独立的、幂等的 POST。

为什么取消是 POST 而不是 DELETE
--------------------------------
取消触发的副作用很重（要落终态、要打断进行中的推理），而且客户端并不拥有
这个资源的生命周期。用 DELETE 会诱使中间的缓存 / 重试中间件对它做危险操作。
POST 的语义就是「我对这件事提一个请求」。

响应体说明
----------
``{"ok": true, "status": "..."}``：

- 任务不存在 → HTTP 404。前端据此提示「这个任务已经结束了」，
  而不是笼统报错——用户点了停止但任务刚好自己跑完了，这是正常时序。
- 已取消 → ``status: "cancelled"``。**幂等**：连点两次停止、
  或前端超时重试，都不该报错。
- 已完成（done / failed / cancelled）→ ``status`` 是原值，**状态不变**。
  刻意不把已完成的任务改成 cancelled：用户点停止的时机晚了 3 秒，
  答案已经在库里并显示出来了，此时把界面改成「已取消」会让用户以为答案丢了。
  停止一个已经完成的任务，正确后果就是「什么都不发生」。

所以 ``ok`` 恒为 true（走到这里说明任务存在），前端该看的是 ``status``。
"""
from fastapi import APIRouter, HTTPException

from app.core.queue import get_queue

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.post("/{task_id}/cancel")
def cancel_task(task_id: str):
    """请求中止一个正在跑的任务。

    立即把任务状态推到 ``cancelled`` 并让 SSE 关流（界面马上停），
    后台执行方在下一个检查点真正退出（见 :class:`app.core.cancel.TaskCancelled`）。
    """
    q = get_queue()
    if not q.cancel(task_id):
        raise HTTPException(404, "任务不存在或已结束")
    # 读回真实状态而不是直接返回 "cancelled"：任务可能已经完成，
    # 此时 cancel 是个 no-op（见 queue.TaskQueue.cancel 的说明），
    # 硬编码 "cancelled" 会让前端把一个已完成的答案误标成被取消。
    status = (q.get_status(task_id) or {}).get("status", "cancelled")
    return {"ok": True, "status": status}
