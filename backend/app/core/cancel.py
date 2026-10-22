"""任务取消信号（中止 / 撤回共用）。

为什么单独一个模块
------------------
取消信号要跨三层传：`TaskQueue` 负责置标志，``ChatAgent`` / ``LLMClient``
负责检查，而这三层互相独立。把异常类型塞进 ``queue.py`` 会让
``app/agent/llm.py`` 为了一个异常类型去 import 队列模块（agent 层依赖
core 层没问题，但方向上会让「队列」变成所有层的公共依赖）；塞进
``chat.py`` 则更糟——``llm.py`` 要 import ``chat``，而 ``chat`` 已经 import
``llm``，直接成环。所以它必须自己占一个模块。

为什么继承 BaseException 而不是 Exception
-----------------------------------------
**这是刻意的，也是这个模块存在的全部理由。**

取消信号会在「阻塞调用正在执行」的任意位置抛出，而本项目里有 6 处以上
``except Exception`` 的宽兜底（``llm._try_once`` 的网络重试、
``llm._stream_once``、``chat._summarize_oldest``、``chat._restream_final``、
``chat._reload_doc``、``chat._load_history``）。它们各自都有正当理由：
宽兜底是为了不让局部故障带崩整条流水线。

但对取消信号来说，**被任何一个 ``except Exception`` 吞掉都是 bug**：
用户点了停止，界面继续转圈直到超时，而日志里什么异常都没有——
一个完全没有信号的中止失败，比中止本身还难查。

逐个加 ``except TaskCancelled: raise`` 需要在 6 个地方各写一遍，
且**第 7 处漏了就是静默失效**，而这种守卫写漏了没有任何测试会红
（本项目已有同类教训：队列事件语义那三个 bug 存在时 348 个测试全绿）。

继承 ``BaseException`` 之后，``except Exception`` 在语言层面就抓不到它，
信号必然一路冒到 :meth:`ChatAgent.run` 的唯一处理点。这与
``KeyboardInterrupt`` / ``SystemExit`` / ``asyncio.CancelledError``
（3.8 之后也是 BaseException）用的是同一套机制，不是奇技淫巧。

代价：``except BaseException`` / 无 except 的裸 ``except`` 会抓到它。
本项目没有这类写法（``finally`` 只做清理，不判断类型），所以代价为零。
"""
from __future__ import annotations


class TaskCancelled(BaseException):
    """用户请求中止当前任务。

    继承 ``BaseException`` 的理由见模块 docstring——简版：取消信号不能被
    任何 ``except Exception`` 吞掉，否则中止会静默失效。

    ``BaseException`` 的子类默认 ``args`` 为空，显式接受一条 message 是为了
    让日志与 SSE 帧里能带上「是谁在什么阶段被取消」这种可读信息。
    """

    def __init__(self, message: str = "任务已取消") -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志可读性
        return self.message
