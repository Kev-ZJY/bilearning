"""撤回的互斥锁：让「删除」与「落库」不可能交错。

这个模块解决的是一个被端到端探针抓到的真 bug
------------------------------------------------
用户撤回一条消息，界面上的消息确实没了，**两秒后聊天区又凭空冒出一条回复**。

同一次运行的两次插桩给出了完全不同的结果：
    第一次：DELETE 只看到 id=1 → removed=1 → id=2 在删除之后出现
    第二次：DELETE 看到 id=1 和 id=2 → removed=2 → 干净
同一份代码、同一个输入，结果取决于两个事务的交错顺序。

为什么「把裁决和写入放进同一个事务」并不足够
--------------------------------------------
第一版修法是把 agent 侧改成「查锚点 + 写回复」共用一个事务，理由是
SQLite 的写锁会串行化两者。**实测证明这个理由是错的**，连跑 5 次探针：
3 次仍失败。插桩出来的真实时序是::

    184.8ms  AGENT   _save_message 进入（拿到写锁）
    188.0ms  CLIENT  DELETE 返回 removed=1      ← 它的 SELECT 跑在 agent 提交之前
    189.0ms  AGENT   _save_message 返回 True    ← agent 提交，回复留在库里

写锁确实是串行的，但**串行不等于「后到者必然后删」**：DELETE 先跑完
（删掉了当时看得见的 1 条）��agent 才提交。回复的 id 比被删的消息大，
可它是在 DELETE 的 SELECT **之后**才产生的，那条 SELECT 根本没见过它。

一句话：**删除只能删掉「删除时已经存在的东西」**。要盖住「删除之后才落库」，
光靠「落库前查一下还在不在」不够 —— 那一刻锚点确实还在（agent 先拿的锁），
守卫正确地放行了，然后 DELETE 才把锚点删掉。谁都没做错，顺序就是错的。

所以正确的不变量不是「落库前查一次」，而是
    **「删除」与「落库」必须互斥，且互斥点必须包住整个删除+裁决+写入。**
只要两者严格串行，「谁先谁后」就只有两种可能，而两种都是干净的：
    - agent 先：回复 id 更大，DELETE 的 ``id >= mid`` 会连它一起删掉
    - DELETE 先：agent 随后裁决，发现锚点已消失，不写

为什么用进程内锁而不是数据库锁
------------------------------
数据库层面做不到「删除操作」本身持锁：``DELETE ... WHERE id >= mid`` 是一���
瞬时语句，语句结束锁就放了，之后才落库的回复照样进得来。要让锁覆盖
「查 + 删」并持续到 agent 的裁决之后，只能有一个跨语句的对象 ——
也就是这里的进程内锁。

**代价要说清楚**：它只在**单个进程**内有效。多副本部署时两个后端进程
各有一把锁，互相看不见。项目当前是单进程（``uvicorn`` 单 worker），
所以这是准确的；真要横向扩 Worker 时，这一段必须换成数据库级的
「撤回墓碑」（给 Message 加 ``recalled_at``，DELETE 打标记而不是删行，
agent 落库时用 ``WHERE NOT EXISTS`` 之类的条件写入）。**不要在没换掉
这把锁之前就扩副本** —— 那会把一个偶发 bug 变成难查的偶发 bug。

本项目是「单进程 + SQLite」，这是准确的边界，写在这里是为了让
将来扩副本的人第一眼就看到，而不是踩进去。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

# 全进程唯一。chat.py（落库）与 api/messages.py（撤回）共用同一把。
_RECALL_LOCK = threading.Lock()


@contextmanager
def recall_guard() -> Iterator[None]:
    """进入撤回临界区。

    用法（两侧必须都包住，否则等于没加）::

        # 删除侧：查 + 删，全程持锁
        with recall_guard():
            with session_scope() as db:
                ...  # 查锚点、删消息

        # 落库侧：裁决 + 写，全程持锁
        with recall_guard():
            with session_scope() as db:
                if db.get(Message, anchor) is None:
                    return False   # 已撤回，不写
                db.add(...)

    锁的范围刻意只包住数据库操作，不含 LLM 调用 ——
    一次工具循环可能几十秒，把锁持那么久会让撤回请求卡在等锁上，
    那是用一个 bug 换另一个 bug。
    """
    _RECALL_LOCK.acquire()
    try:
        yield
    finally:
        _RECALL_LOCK.release()
