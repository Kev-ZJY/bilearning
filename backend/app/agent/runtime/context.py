"""对话提示词、输入 token 粗估与历史裁剪；不依赖数据库和任务状态。"""
import json
from app.agent.runtime.model_limits import DEFAULT_INPUT_BUDGET_TOKENS

# 输入侧估算预算，与 LLM max_tokens 输出预算分别管理。
CONTEXT_BUDGET_TOKENS = DEFAULT_INPUT_BUDGET_TOKENS


def document_for_model(doc):
    """Keep knowledge and figure references; omit stored binary data and audit logs.

    Image bytes are only sent as actual image inputs. Serialising base64 into
    prose can exhaust the context on a second video in the same project.
    """
    if isinstance(doc, list):
        return [document_for_model(value) for value in doc]
    if not isinstance(doc, dict):
        return doc
    result = {}
    for key, value in doc.items():
        if key in ("verification", "visual_review", "independent_review", "reviews", "illustrations"):
            continue
        if key == "src" and isinstance(value, str) and value.startswith("data:image/"):
            result[key] = "stored-figure:" + str(doc.get("id", ""))
        else:
            result[key] = document_for_model(value)
    return result

def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数（中文按字、英文按 4 字符）。"""
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        # CJK 统一表意文字 + 扩展区 + 中日韩标点
        if "一" <= ch <= "鿿" or "　" <= ch <= "〿":
            cjk += 1
        else:
            other += 1
    return cjk + other // 4


def _project_image_payloads(value):
    """Replace inline image data with a marker and return its image count."""
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return "[image payload omitted]", 1
        return value, 0
    if isinstance(value, list):
        projected = []
        count = 0
        for item in value:
            child, child_count = _project_image_payloads(item)
            projected.append(child)
            count += child_count
        return projected, count
    if isinstance(value, dict):
        projected = {}
        count = 0
        for key, item in value.items():
            child, child_count = _project_image_payloads(item)
            projected[key] = child
            count += child_count
        return projected, count
    return value, 0


def _estimate_value_tokens(value, *, image_token_cost=768) -> int:
    projected, image_count = _project_image_payloads(value)
    if isinstance(projected, str):
        text_cost = estimate_tokens(projected)
    else:
        text_cost = estimate_tokens(json.dumps(projected, ensure_ascii=False))
    return text_cost + image_count * max(0, int(image_token_cost))


def estimate_llm_input_tokens(
    messages: list[dict],
    *,
    tool_schemas=None,
    extra_payload=None,
    image_token_cost: int = 768,
) -> int:
    """Estimate complete request input without treating image base64 as prose."""
    image_token_cost = max(0, int(image_token_cost))
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            total += _estimate_value_tokens(message, image_token_cost=image_token_cost) + 4
            continue
        total += _estimate_value_tokens(
            message.get("content") or "", image_token_cost=image_token_cost,
        ) + 4
        metadata = {
            key: value for key, value in message.items()
            if key not in ("role", "content")
        }
        if metadata:
            total += _estimate_value_tokens(metadata, image_token_cost=image_token_cost)
    if tool_schemas is not None:
        total += _estimate_value_tokens(tool_schemas, image_token_cost=image_token_cost)
    if extra_payload:
        total += _estimate_value_tokens(extra_payload, image_token_cost=image_token_cost)
    return total


def _system_prompt(project_title: str, has_doc: bool,
                   focus: str = "") -> str:
    """构造系统提示。

    两条硬约束写在这里：**不许编造**、**不许用元叙述**。
    前者是知识类助手最容易犯的错（检索没命中就自己编一段听起来合理的
    解释，用户完全看不出来）；后者是本项目写作规范的延续——项目记忆里
    明确要求产出内容不得出现「讲师」「黑板」「视频中」这类元叙述。

    ``focus`` 是用户此刻在右侧面板的浏览位置（已格式化成
    「《文档》› 模块 › 知识点」这样的路径串，可为空）。

    **为什么要传它**：用户报的真实问题是——他第 4 轮说「我说的 html 要改为
    Apple 风格」，助手却反问「是苹果公司、苹果产品、苹果网站应用，还是一个网页？」。
    而他第 1 轮明明说过是做 iPhone/Mac 的产品。两层原因叠加：

    1. 后端完全不知道用户正对着哪份文档的哪一块，模型只能靠对话猜作用域；
    2. 早期对话已被 ``MAX_HISTORY`` 按条数截断，那句「iPhone、Mac」早就不在了。

    第 1 条是根因，第 2 条让它无法被对话兜住。模型在信息缺失时反问是**合理**的，
    它不是笨，是架构没把信息送到 —— **该进上下文的信息没进，
    后面再好的压缩与摘要也救不回来。**

    焦点为空时这一整段不出现。不能写「用户未指定正在查看的位置」：
    那等于明确告诉模型「这里缺信息」，反而诱发它反问。
    """
    progress_hint = "调用工具前用一两句公开进度说明告诉用户要检查什么；不要输出私密推理。工具结束后继续执行，最终回复直接回答用户，不重复中间日志。知识文档已在右栏，不生成额外交付物打开组件。\n"
    doc_hint = progress_hint + (
        "当前项目已经整理过视频，右侧有知识文档。"
        "回答用户关于文档内容的问题前，**先用工具检索**，不要凭印象作答。"
        if has_doc
        else "当前项目还没有整理过任何视频，知识文档是空的。"
             "用户问文档内容时，直接告诉他需要先粘贴 B 站链接整理视频，"
             "**不要凭空编造知识点**。"
    )
    focus_hint = ""
    if focus.strip():
        focus_hint = f"""
## 用户此刻在看哪里

用户当前正在看：**{focus}**

用户说「这个」「这里」「改一下」时，指的就是上面这处，**直接按这个作用域理解**。
他说的「html」「样式」「排版」指的是右侧文档的呈现形态，不是另一个文件。

**不要因为信息已经给足还反问作用域。** 用户报过的真实反例：他前几轮已经说明
在做 iPhone/Mac 相关的产品，第 4 轮要求「把 html 改成 Apple 风格」，
助手却反问「是苹果公司、苹果产品、苹果网站应用，还是一个网页」——
这是把用户已经给过的信息再问一遍。判断依据是信息**在不在上下文里**：
不在就问，在就动手。
"""
    return f"""你是「视频学习工作台」的整理助手，正在项目「{project_title}」里和用户对话。

你的能力：检索知识文档、解释知识点、查看视频列表、触发视频整理、按用户要求改写文档。

## 当前状态
{doc_hint}
{focus_hint}
## 硬性要求

**不许编造**：知识类问题必须基于工具检索到的内容回答。检索没命中时，
直接告诉用户「文档里没有这部分内容」，并说明文档里有什么（可先调
read_outline）。**编造出来的知识点比不回答危害大得多**——用户分辨不出来。

**引用要具体**：说到某个知识点时带上它的模块名或标题，让用户能在右侧
面板里找到它。

**提交任务后不要等**：调 submit_process 之后任务在后台跑几分钟到几十分钟，
你要告诉用户「已提交、稍后右侧面板会更新」，**不要说「已经整理好了」**——
文档此刻还没生成。

**不要用元叙述**：不要写「视频里讲师说」「这段视频讲了」这类描述。
你在回答用户的问题，不是在转述视频。

**改写要最小改动**：调 rewrite_document 时只改用户点名的部分，
不要顺手重写整个文档。
"""


def _messages_tokens(messages: list[dict]) -> int:
    return sum(_estimate_value_tokens(m.get("content") or "") for m in messages)


def validate_messages_budget(*, messages: list[dict], tool_schemas: list[dict],
                             budget: int, extra_token_cost: int = 0) -> int:
    """校验一次实际模型调用的完整输入，包括工具回灌与非文本预留。"""
    total = estimate_llm_input_tokens(
        messages, tool_schemas=tool_schemas,
    ) + max(0, int(extra_token_cost))
    if total > budget:
        raise ContextBudgetError(
            f"本轮模型输入估算为 {total} token，超出预算 {budget}；"
            "包含历史、当前问题、工具调用/结果、文档与图像预留。"
        )
    return total


def estimate_document_tokens(doc: object, *, image_token_cost: int = 768) -> int:
    """估算当前项目知识文档与图像输入的上下文预留。

    文档按实际可提供给工具/模型的文本投影估算；base64 图片不序列化计数，
    每张已存图像另预留固定估值。不同模型的视觉计费会变化，这只是保守预算，
    不是服务端 tokenizer 的精确计数。
    """
    image_count = 0

    def count_images(value: object) -> None:
        nonlocal image_count
        if isinstance(value, dict):
            src = value.get("src")
            if isinstance(src, str) and src.startswith("data:image/"):
                image_count += 1
            for child in value.values():
                count_images(child)
        elif isinstance(value, list):
            for child in value:
                count_images(child)

    count_images(doc)
    projected = document_for_model(doc)
    text_cost = _estimate_value_tokens(projected)
    return text_cost + image_count * max(0, int(image_token_cost))


def assemble_messages(*, system_text: str, history: list[dict],
                      current_text: str, tool_schemas: list[dict],
                      budget: int, summarize=None,
                      summary_message: str = "",
                      preserve_history: bool = False,
                      extra_token_cost: int = 0) -> tuple[list[dict], int, int]:
    """按 token 预算把「系统提示 + 历史 + 当前问题」组装成 messages。

    返回 ``(messages, kept, dropped)``：kept 是实际装进去的历史条数，
    dropped 是被压缩掉的条数（0 表示没压缩）。

    为什么按预算而不是按条数
    ------------------------
    条数与 token 没有稳定关系：10 条闲聊可能几百 token，10 条含文档改写的
    讨论能到几万。唯一的可靠控制是 token。而且**工具 schema 本身也要算**——
    它的 JSON 常常上万 token，早先完全没计入，于是「历史只放了 10 条」看起来
    很小，实际请求早就超了。

    为什么压缩要「锚定」
    ------------------
    Anthropic 在《Effective context engineering》里点了压缩的尖锐失败模式：
    **摘要里漏掉的关键事实，比没有摘要更糟** —— 因为后续步骤会信任这份摘要，
    已经无法回到原文去查。对策是 anchored summarization：把必须原样保留的
    部分（这里是系统提示与当前问题）**逐字保留、绝不摘要**，只压缩中间的对话。

    压缩失败时怎么办
    --------------
    刻意**不静默降级**。摘要是额外的 LLM 调用，它可能超时或失败；
    这时把原始历史留着（少几条）远好过让整条对话崩掉，所以失败即放弃压缩。
    但放弃压缩后如果仍然超预算，必须显式告知 —— 静默超预算等于把问题
    推迟到 API 拒绝那一刻才暴露，那时用户什么都拿不到。

    整轮工具循环的累积
    ------------------
    注意这里只预算了「进入循环前」的上下文。每跑一轮工具调用，
    该轮的工具结果会再加进 messages 继续累积（这才是真正的膨胀源）。
    那部分由 MAX_TOOL_ROUNDS 间接兜住（轮数有上限），不再另做预算 ——
    超过就如实报错，不做静默截断。
    """
    head: list[dict] = [{"role": "system", "content": system_text}]
    if summary_message:
        # Durable conversation summaries are sourced historical data. Keep them
        # in the transcript as an assistant history item, never as system policy.
        head.append({"role": "assistant", "content": summary_message})
    tail: list[dict] = [{"role": "user", "content": current_text}]

    # 工具 schema 的体积必须计入预算：它是固定开销，且常常不小。
    schema_cost = _estimate_value_tokens(tool_schemas)

    extra_token_cost = max(0, int(extra_token_cost))
    used = (_messages_tokens(head) + _messages_tokens(tail) + schema_cost
            + extra_token_cost)
    remaining = budget - used
    if remaining < 0:
        raise ContextBudgetError(
            f"固定部分（系统提示 + 工具定义 + 当前问题）已占 {used} token，"
            f"超出预算 {budget}。这通常意味着单条系统提示或工具定义异常膨胀，"
            f"不是对话历史的问题。")

    if preserve_history:
        history_cost = _messages_tokens(history)
        total = used + history_cost
        if total > budget:
            raise ContextBudgetError(
                f"固定输入、文档/图像预留与最近 {len(history)} 条历史原文共需 "
                f"{total} token，超出预算 {budget}；不能缩短承诺保留的最近 10 个用户轮次。"
            )
        return head + list(history) + tail, len(history), 0

    # 从近往远装：最近的一轮最重要，用户当下的问题往往依赖它。
    kept_rev: list[dict] = []
    for m in reversed(history):
        cost = _estimate_value_tokens(m.get("content") or "")
        if cost > remaining:
            break
        remaining -= cost
        kept_rev.append(m)
    kept = list(reversed(kept_rev))
    dropped = len(history) - len(kept)

    if dropped and summarize is not None:
        older = history[:len(history) - len(kept)]
        summary = summarize(older)
        if summary:
            # 摘要以**系统角色**插入在 head 之后：它是「关于早期对话的说明」，
            # 不是一轮真实对话。用 user 角色会被模型当成用户说的话。
            head.append({"role": "system", "content": summary})
            if _messages_tokens(head) + _messages_tokens(kept) + _messages_tokens(tail) + schema_cost + extra_token_cost > budget:
                raise ContextBudgetError(
                    f"历史摘要插入后上下文超出预算 {budget} token。"
                )
            dropped = 0  # 摘要成功就不算「丢弃」了
        else:
            # 压缩失败：保持原始历史（少几条），但明确告知超预算这件事。
            raise ContextBudgetError(
                f"保留了最近 {len(kept)} 条对话仍超出预算 {budget} token，"
                f"且压缩早期对话失败。请开一个新对话，或缩短单条消息。")

    return head + kept + tail, len(kept), dropped


class ContextBudgetError(RuntimeError):
    """组装后的上下文超出 token 预算。

    独立成一个类型而不是直接抛 RuntimeError：调用方要能只捕获这一类，
    避免把「上下文装不下」和「LLM 调不通」混成同一个错误处理。
    """


_COMPACT_INSTRUCTIONS = (
    "你在压缩一段 agent 与用户的早期对话。写一份高保真的摘要，"
    "让后续对话能无缝继续，**不要重新开始一段对话**。\n\n"
    "必须逐条保留（这些丢了就无法从别处恢复）：\n"
    "1. 用户的目标与意图（他在做什么、要什么结果）\n"
    "2. 用户明确提出过的硬性约束、偏好、否定项（他说过「不要 X」就一定要留着）\n"
    "3. 已经达成的结论与已经做过的决定\n"
    "4. 尚未解决的问题与用户提出的未决疑问\n"
    "5. 涉及的具体名词、模块名、标题、专有名词（逐字保留，不要意译）\n\n"
    "可以丢弃：寒暄、过程性描述、已被后续内容推翻的中间尝试、重复表述。"
    "用中文，控制在 600 字以内，直接输出摘要正文，不要加任何前言或标题。"
)
