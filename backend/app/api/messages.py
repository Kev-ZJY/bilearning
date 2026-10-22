import json
"""消息接口：列出 / 追加用户消息，并触发 agent 产生回复。

这一轮修的是本项目最大的功能空洞
----------------------------------
过去 ``create_message`` 只把用户消息落库就 ``return``，文件头注释还明确写着
「助手回复由流水线里的 LLM 调用产生，不在这个端点里」。结果是：用户在对话框里
发任何非链接内容，消息出现在聊天区，然后**永远没有任何回复**。产品叫「助手」
但实际只是个链接下载器 + 进度条。

现在的流程
----------
1. 用户消息落库（role="user"）
2. 提交一个 ``kind="chat"`` 的任务到队列
3. 起后台线程跑 :class:`~app.agent.conversation.chat.ChatAgent`（工具循环 + 落库回复）
4. 立即返回 ``{message, task_id}``，前端拿 task_id 连 SSE 看进度

为什么必须异步
--------------
``ChatAgent`` 里的 ``llm.chat_tools`` 是阻塞调用（实测单轮 6-24 秒，工具循环
2-3 轮）。在请求线程里同步跑会让 HTTP 请求挂住几十秒，且期间发不出任何
进度事件——这正是「进度条完全不动」那个被投诉两次的问题。

为什么 task_id 要和视频流水线区分
---------------------------------
两者共用同一个任务队列与 ``/api/tasks/{id}/events`` 通道。对话 id 走
``chat-`` 前缀，与流水线的 ``task-{video_id}-{ts}`` 天然不撞。撞车的话
两条消息会共用一个进度条，用户看到的是「两条消息的进度混在一起」。

对话的 stage 是 ``"chat"``，**刻意不在** ``pipeline._STATUS_SYNC_STAGES``
里：那个集合里的阶段名会被写进 ``Video.status``，套用会污染视频卡片上的
状态标签（前端只映射了流水线那六个枚举）。

本模块还提供**撤回**端点（``DELETE /{pid}/messages/{mid}``），
它和中止（``app/api/tasks.py`` 的 cancel）是两件事：
中止是「停下来，但这一问还在」，撤回是「这一问不算数，删掉重来」。
"""

import threading
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.conversation.chat import ChatAgent

router = APIRouter(prefix="/api/projects", tags=["messages"])


class FocusRef(BaseModel):
    """用户此刻正在看知识文档的哪一处。

    **为什么需要它**（用户报的真实问题）：
    右侧面板始终显示当前项目的知识文档，但后端完全不知道用户看的是哪一块。
    于是用户说「我说的 html 要改为 Apple 风格」时，模型只能靠对话猜作用域，
    反问「是苹果公司、苹果产品、苹果网站应用，还是一个网页？」——
    而用户前几轮明明已经说过是做 iPhone/Mac 的产品。

    模型在拿不到信息时反问是**合理**的。它不是笨，是架构没把信息送到。
    这属于 context engineering 里最基础的一层：该进上下文的信息没进，
    后面再好的压缩与摘要也救不回来。

    字段说明：
    - ``doc_title``：文档标题（meta.title），用于消歧「哪一份文档」
    - ``point_title``：用户当前展开/停留的知识点标题
    - ``module_title``：该知识点所属模块，用于给出完整路径
    - ``view``：当前视图形态（html / md / pdf）。用户说「改 html」时，
      模型必须知道 html 是当前视图而不是「另一个文件」
    """

    doc_title: str = ""
    point_title: str = ""
    module_title: str = ""
    view: str = ""


class MessageCreate(BaseModel):
    content: str
    video_ids: list[int] = Field(default_factory=list, max_length=100)
    # 可选：用户此刻的浏览位置。为空时按「未指定焦点」处理，
    # 系统提示词里那一段就不出现（而不是写成「未指定」——那会提示模型
    # 「你可能缺信息」，诱发它反问）。
    focus: Optional[FocusRef] = None


def _serialize(m) -> dict:
    try:
        saved = json.loads(m.tool_calls_json or "{}")
        activity = saved.get("activity") if isinstance(saved, dict) else None
    except (ValueError, TypeError):
        activity = None
    return {
        "id": m.id,
        "project_id": m.project_id,
        "role": m.role,
        "content": m.content,
        "activity": activity,
        "created_at": m.created_at.isoformat() if m.created_at is not None else None,
    }


@router.delete("/{pid}/messages/{mid}")
def delete_message(pid: int, mid: int):
    """撤回一条消息及其之后的全部回复。

    **「撤回」的产品定义**（用户给的原话）：中止当前会话、内容重回输入框、
    会话历史里不再包含当前这条 query。所以这个端点做三件事：

    1. 取消仍在跑的任务（否则 agent 会在删完之后把回复又落回来，
       界面上「撤回后凭空又冒出一条回复」）。
    2. 删掉该条消息**以及它之后的所有消息**。只删自己一条是不够的：
       一问一答是两条记录，删掉问、留着答，历史就变成了一段无头的独白，
       下一轮模型读到它会莫名其妙。
    3. **返回被删的用户消息原文**，让前端把它填回输入框——用户改一个字就能重发，
       不用重新手打。这一步是「撤回」与「删除」的本质区别。

    删的范围用 ``id >= mid`` 而不是 ``created_at >``：时间戳可能相同
    （同一秒内落库），按时间比会漏掉同时刻的后续消息。
    """
    from app.core.database import session_scope
    from app.core.queue import get_queue
    from app.core.recall import recall_guard
    from app.models.entities import Message, Project

    # 先在锁外把要还原的原文和 task_id 取出来（下面要重新查一次库），
    # 这一段只读，不必占锁。
    with session_scope() as db:
        if not db.get(Project, pid):
            raise HTTPException(404, "项目不存在")
        m = db.get(Message, mid)
        if m is None or m.project_id != pid:
            raise HTTPException(404, "消息不存在")

        content = m.content
        task_id = get_queue().find_by_message(mid)

    # ---- 先取消，再删除 ----
    #
    # 顺序不能反，但**真正保证正确性的不是顺序，是下面那把锁**。
    # 实测踩过的坑：以为「查+删」和 agent 的「裁决+写」各自在一个事务里就
    # 够了 —— 不够。插桩出来的真实时序是
    #     184.8ms  agent 拿到写锁、开始裁决
    #     188.0ms  这里的 DELETE 返回 removed=1（它的 SELECT 早于 agent 提交）
    #     189.0ms  agent 提交，回复留在库里
    # 写锁确实串行了，但**串行 ≠ 后到者必然后删**：这里的 SELECT 跑在 agent
    # 提交之前，那条回复当时根本还不存在，自然删不到。
    #
    # 「删除」只能删掉「删除时已经存在的东西」。所以必须让两侧互斥：
    # 拿到 recall_guard 之后只有两种顺序，而两种都是干净的 ——
    #   agent 先：回复 id 更大，``id >= mid`` 会连它一起删掉
    #   这里先：agent 随后裁决，发现锚点已消失，不写
    #
    # 取消放在锁外、删除之前，是为了让线程尽快停下、少烧一点 token，
    # 不承担正确性责任。
    if task_id:
        get_queue().cancel(task_id)

    with recall_guard(), session_scope() as db:
        doomed = (
            db.query(Message).filter(Message.project_id == pid, Message.id >= mid).all()
        )
        removed = len(doomed)
        for row in doomed:
            db.delete(row)

    return {"ok": True, "restored": content, "removed": removed}


def _new_chat_task_id() -> str:
    """生成对话任务 id。

    带 ``uuid4`` 后缀而不是只用秒级时间戳：同一秒内连发两条消息
    （用户快速提问、或前端重试）会撞 id，而撞 id 意味着两条消息共用
    一个任务状态——界面上表现为「两条消息的进度混在一条进度条里」。
    """
    return f"chat-{int(time.time())}-{uuid.uuid4().hex[:8]}"


# 视图名 → 给模型看的中文。模型不知道「html 视图」是我们的界面概念，
# 直接把英文枚举塞进提示词，它会当成文件名或格式关键字去猜。
_VIEW_LABELS = {
    "html": "右侧 HTML 视图",
    "md": "右侧 Markdown 视图",
    "pdf": "右侧 PDF 视图",
}


def _format_focus(focus: Optional[FocusRef]) -> str:
    """把浏览位置拼成一句给模型看的话。

    返回空串表示「用户没指定位置」——**这时候整段都不该出现在提示词里**。
    写一句「用户未指定正在查看的位置」反而是坏主意：它明确告诉模型
    「这里缺信息」，于是模型就会反问，正是我们要消灭的那种行为。

    只在真的有内容时才拼，且按「文档 › 模块 › 知识点」的路径顺序，
    因为模型需要的是**可直接引用的定位串**（用户说的「这个模块」指哪，
    模型靠它对号入座）。
    """
    if focus is None:
        return ""
    parts: list[str] = []
    if focus.doc_title.strip():
        parts.append(f"《{focus.doc_title.strip()}》")
    if focus.module_title.strip():
        parts.append(focus.module_title.strip())
    if focus.point_title.strip():
        parts.append(focus.point_title.strip())
    if not parts:
        return ""
    where = " › ".join(parts)
    view = _VIEW_LABELS.get((focus.view or "").strip().lower())
    if view:
        where = f"{where}（当前显示在{view}）"
    return where


@router.get("/{pid}/messages")
def list_messages(pid: int):
    from app.core.database import session_scope
    from app.models.entities import Project, Message

    with session_scope() as db:
        p = db.get(Project, pid)
        if not p:
            raise HTTPException(404, "项目不存在")
        msgs = (
            db.query(Message)
            .filter_by(project_id=pid)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .all()
        )
        return [_serialize(m) for m in msgs]


@router.post("/{pid}/messages")
def create_message(pid: int, body: MessageCreate):
    """追加用户消息并触发 agent 回复（异步，进度走 SSE）。

    返回 ``{message, task_id}``：``message`` 是刚落库的用户消息，
    ``task_id`` 用来订阅 ``/api/tasks/{task_id}/events``。
    """
    from app.core.database import session_scope
    from app.core.queue import get_queue
    from app.models.entities import Message, Project, Video

    content = (body.content or "").strip()
    if not content:
        raise HTTPException(400, "消息内容不能为空")

    # 焦点信息在这里就规整成「给模型看的字符串」，
    # 别把 pydantic 对象带进后台线程：模型提示词只该拿到最终文案。
    focus = _format_focus(body.focus)

    with session_scope() as db:
        p = db.get(Project, pid)
        if not p:
            raise HTTPException(404, "项目不存在")
        if body.video_ids:
            attached = (
                db.query(Video)
                .filter(Video.project_id == pid, Video.id.in_(body.video_ids))
                .all()
            )
            if len(attached) != len(set(body.video_ids)):
                raise HTTPException(400, "本轮视频不属于当前项目")
            focus += "\n本轮用户提交的视频 ID：" + ", ".join(
                str(v.id) for v in attached
            )
            focus += "。按用户文字要求处理这些视频；仅提交链接时默认整理并融合知识点。"
        m = Message(project_id=pid, role="user", content=content)
        db.add(m)
        db.flush()  # 需要在 commit 之前拿到自增 id 返回给前端

        # 把 agent 需要的上下文在这里一并取出：后台线程不该再开一次
        # session 查库（多线程下 session 不是线程安全的共享对象）。
        doc: dict = {}
        if p.doc and p.doc.doc_json:
            import json

            try:
                doc = json.loads(p.doc.doc_json)
            except json.JSONDecodeError:
                doc = {}
        videos = [
            {
                "id": v.id,
                "bvid": v.bvid,
                "cid": v.cid,
                "title": v.title,
                "status": v.status,
                "duration": v.duration,
            }
            for v in db.query(Video).filter_by(project_id=pid).all()
        ]
        project_title = p.title
        storage_root = p.storage_root
        payload_msg = _serialize(m)

    task_id = _new_chat_task_id()

    # 先登记状态供 SSE 订阅，再启动本轮对话线程。
    get_queue().register(
        task_id,
        {
            "kind": "chat",
            "project_id": pid,
            "message_id": payload_msg["id"],
        },
    )

    def work() -> None:
        # ChatAgent 内部自己处理异常并落库/回写状态，这里只做最后兜底：
        # 万一它自己都崩了（比如数据库不可用），不能让任务悬在 running，
        # 否则前端进度条永远转下去。
        try:
            ChatAgent(
                pid,
                doc=doc,
                videos=videos,
                project_title=project_title,
                storage_root=storage_root,
                focus=focus,
                input_video_ids=body.video_ids,
                # 把「是哪条消息触发了这次对话」交给 agent。
                # 它在落库前会用这个 id 做一次裁决：消息已经不在了
                # （= 用户撤回了）就不写。见 ChatAgent._save_message。
                anchor_message_id=payload_msg["id"],
            ).run(task_id, content)
        except Exception as e:
            get_queue().update(
                task_id, status="failed", error=str(e), message=f"对话处理失败：{e}"
            )

    threading.Thread(target=work, daemon=True).start()
    return {"message": payload_msg, "task_id": task_id}
