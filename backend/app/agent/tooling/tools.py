"""Agent 工具：按需检索知识文档、查看视频、提交整理、写回改写结果。

检索结果必须显式标注截断；空文档返回可解释结果；同名 ID 返回全部匹配。
有副作用的整理操作统一委托给 services.processing。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

# Compatibility exports: query imports and ToolRegistry dispatch patches remain supported.
# Private query helpers now resolve their globals in knowledge_queries.
from app.agent.tooling.knowledge_queries import (
    MAX_SEARCH_CHARS, _fmt_duration, _iter_points,
    get_point, list_videos, read_outline, search_knowledge,
)
from app.agent.tooling.tool_definitions import chat_tool_schemas


# ---------------------------------------------------------------------------
# 有副作用的工具：提交流水线 / 改写文档
# ---------------------------------------------------------------------------

def submit_process(project_id: int, video_id: int, intent: str = "整理视频知识点",
                   terms: Optional[list[str]] = None, options: dict | None = None) -> dict:
    """触发视频整理流水线（复用现有 Pipeline，进度走 SSE）。

    这里只**提交**不等待：流水线本身要跑几分钟到几十分钟，
    agent 不该在这里阻塞。等它完成后用户会在右侧面板看到文档更新。
    """
    from app.services.processing import ProcessingNotFound, submit_processing

    try:
        task_id = submit_processing(project_id, video_id, intent, terms, options)
    except ProcessingNotFound as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "task_id": task_id,
        "video_id": video_id,
        "note": "仅已提交，尚未完成。必须依据任务实际状态说明结果；不能承诺文档一定更新。",
    }


def update_knowledge(project_id: int, doc: dict, *, expected=None, task_id=None) -> dict:
    """共享门禁、并发基线检查及版本留存；失败不写入。"""
    from app.services.documents import replace_doc
    try:
        candidate = replace_doc(project_id, doc, expected=expected, task_id=task_id)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    doc = candidate
    n_modules = len(doc.get("modules") or [])
    n_points = sum(len(m.get("points") or [])
                   for m in (doc.get("modules") or [])
                   if isinstance(m, dict))
    return {
        "ok": True,
        "module_count": n_modules,
        "point_count": n_points,
        "note": f"文档已更新（{n_modules} 个模块 / {n_points} 个知识点），"
                "右侧面板已刷新。请在回复里说明你具体改了什么。",
    }


# ---------------------------------------------------------------------------
# 工具注册表：给模型的 schema + 按名调度
# ---------------------------------------------------------------------------

class ToolRegistry:
    """把工具绑到当前项目的上下文上，负责按名调度并记录调用历史。

    为什么要注册表而不是散装函数：agent 循环要 (a) 拿到 schema 发给模型、
    (b) 收到 tool_calls 后按名字执行、(c) 知道这一轮调过什么。绑在一个对象上
    就不用在循环里到处传 project_id / doc / videos。
    """

    def __init__(self, doc: Optional[dict] = None,
                 videos: Optional[list[dict]] = None,
                 project_id: Optional[int] = None,
                 on_update: Optional[Callable[[], None]] = None, task_id=None,
                 historical_context: Optional[dict] = None,
                 current_request: Optional[str] = None):
        self.doc = doc if isinstance(doc, dict) else {}
        self.videos = videos or []
        self.project_id = project_id
        self.task_id = task_id
        # These values are set by ChatAgent from the anchored program snapshot.
        # They never appear in a model-visible tool schema.
        self.historical_context = historical_context
        self.current_request = current_request
        # 文档可能被 update_knowledge 改掉，用后置回调让调用方能重新加载
        self._on_update = on_update
        self.history: list[dict] = []

    # -- schema ----------------------------------------------------------
    def schemas(self) -> list[dict]:
        """Return a fresh chat protocol; ChatAgent owns rewrite dispatch."""
        return chat_tool_schemas()

    # -- 调度 ------------------------------------------------------------
    def run(self, name: str, args: Optional[dict] = None) -> dict:
        """按名执行一个工具，返回带 ``ok`` 的结果字典。

        **不抛异常**：未知工具名、缺参数、工具内部出错，一律返回
        ``{"ok": False, "error": ...}``。原因是模型会看到返回值并自我纠正，
        而抛异常会终止整个 agent 循环——用户看到的就是「agent 不回话了」，
        正是本项目要修的那个症状。
        """
        args = args if isinstance(args, dict) else {}
        self.history.append({"tool": name, "args": args})

        try:
            if name == "read_outline":
                return _ok(read_outline(self.doc))
            if name == "search_knowledge":
                return _ok(search_knowledge(self.doc, args.get("query", "")))
            if name == "get_point":
                return _ok(get_point(self.doc, args.get("point_id", ""),
                                     args.get("module_id")))
            if name == "list_videos":
                return _ok(list_videos(self.videos))
            if name == "submit_process":
                vid = args.get("video_id")
                if not isinstance(vid, int):
                    return {"ok": False,
                            "error": "submit_process 需要整数 video_id，"
                                     f"收到的是 {vid!r}。请先 list_videos 拿 id。"}
                if self.project_id is None:
                    return {"ok": False, "error": "当前没有项目上下文，无法提交任务"}
                raw_options = args.get("options")
                if raw_options is None:
                    raw_options = {}
                if not isinstance(raw_options, dict):
                    return {"ok": False, "error": "submit_process 的 options 必须是对象"}
                # The program's anchored source snapshot is authoritative. A
                # model cannot forge or trim its summary/recent turns by adding
                # a same-named nested option to a tool call.
                from app.services.options import (
                    ProcessingOptions, resolve_current_request_options,
                )
                options = dict(raw_options)
                options = resolve_current_request_options(
                    options, self.current_request or "",
                )
                options["historical_context"] = self.historical_context
                normalized = ProcessingOptions(**options).model_dump()
                intent = self.current_request
                if intent is None:
                    intent = args.get("intent") or "整理视频知识点"
                return submit_process(self.project_id, vid,
                                      intent, options=normalized)
            if name == "update_knowledge":
                if self.project_id is None:
                    return {"ok": False, "error": "当前没有项目上下文，无法写入文档"}
                res = update_knowledge(self.project_id, args.get("doc"), expected=self.doc, task_id=self.task_id)
                if res.get("ok") and self._on_update:
                    self._on_update()
                return res
        except Exception as e:  # 工具内部出错也不许终止循环
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        return {
            "ok": False,
            "error": f"没有名为 {name!r} 的工具。可用工具："
                     "read_outline / search_knowledge / get_point / list_videos "
                     "/ submit_process / update_knowledge",
        }


def _ok(result: dict) -> dict:
    """给纯函数工具的结果套上 ok 信封。"""
    if "ok" in result:
        return result
    out = {"ok": result.get("found") is not False}
    out.update(result)
    return out


def run_tool(name: str, args: Optional[dict] = None, **ctx) -> dict:
    """模块级便捷函数：单测某个工具时不必先建注册表。"""
    return ToolRegistry(**ctx).run(name, args)
