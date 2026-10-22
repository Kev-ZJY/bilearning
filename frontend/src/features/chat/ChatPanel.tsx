import { useEffect, useRef, useState } from "react";
import { marked } from "marked";
import type { AgentStep, Message, Video, TaskEvent } from "../../api/types";
import { AgentActivity } from "../tasks/AgentActivity";

// Video.status 随真实流水线阶段更新；刷新后继续显示补图、审查或修订状态。
const VIDEO_STATUS_LABELS: Record<string, string> = {
  cancelled: "已取消",
  pending: "待整理",
  downloading: "下载中",
  asr: "转写中",
  draft: "抽取知识点中",
  frames: "音画联合分析中",
  verify: "核对来源与覆盖中",
  illustrate: "补充图例中",
  review: "独立审查中",
  revise: "修订知识文档中",
  render: "生成文档中",
  done: "已完成",
  failed: "失败",
  skipped: "已跳过",
};

export function videoStatusLabel(status?: string): string {
  if (!status) return "";
  return VIDEO_STATUS_LABELS[status] ?? status;
}

// 助手回复的 Markdown → HTML。
//
// 为什么必须渲染：模型输出的是 Markdown（系统提示要求它用小标题和列表组织
// 答案），而这里原本是 `{m.content}` 纯文本直出，于是界面上赫然显示着
// `## 先说清楚` 和 `**跨进程调用**` —— 端到端截图抓到的真实观感问题。
// 用户看到符号而不是结构，第一反应是「这答案怎么这么难读」。
//
// XSS 防线（这里踩过一个真实的版本坑）：
// 最初以为 marked 默认会转义 HTML，实测 marked@18 **不会** ——
// `renderMarkdown("正常文本<script>alert(1)</script>")` 返回的字符串里
// 真的带着 `<script>` 标签（v8 之后 sanitize 选项被移除）。而这份内容
// 最终要经 dangerouslySetInnerHTML 注入 DOM，等于开了个真实漏洞。
// 所以这里**先自己把原始 HTML 转义掉再交给 marked**：Markdown 的语法
// 字符（* # - [ ] ` >）里不含 < > &，预先转义不会破坏任何标记语法。
export function renderMarkdown(text: string): string {
  if (!text || !text.trim()) return "";
  // & 必须第一个转，否则会把刚转义出的 &lt; 再转成 &amp;lt;
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return marked.parse(escaped, { async: false, gfm: true, breaks: true }) as string;
}

// 分P 元信息 → 给用户看的明文。后端 addVideo 返回体里带了 page / total_pages /
// page_warning，但前端此前完全没消费（缺陷 2）：用户粘 ?p=999 被回退到 P1 时
// 界面毫无提示，最后拿到完全不相干的视频。这里把信息格式化成展示文案：
// - page_warning 必有（意味着「你要的没拿到」），用 ⚠ 醒目标注；
// - 多P 视频则告知导入了第几集 / 共几集。
// 写法与 VIDEO_STATUS_LABELS 同风格：纯函数、无副作用、空值安全。
export function formatVideoPageInfo(video: Video): string {
  if (video.page_warning) {
    return `⚠ ${video.page_warning}`;
  }
  if (video.total_pages && video.total_pages > 1) {
    const p = video.page ?? 1;
    return `已导入第 ${p} 集 / 共 ${video.total_pages} 集`;
  }
  return "";
}

/**
 * 这一下回车该不该被当成「发送 / 提交」。
 *
 * 中文输入法下用拼音敲一个英文单词（比如 `zhifu`），选词窗口还开着时
 * 按回车是**确认候选词**，字符还没进输入框。原先只判 `e.key === "Enter" && !e.shiftKey`，
 * 于是这一下直接被发送 —— 表现是「我还没打完就发出去了」，
 * 而且输入框被清空，这段拼音直接丢了。用户报的是这个。
 *
 * `KeyboardEvent.isComposing` 就是浏览器给出的「正在组词」标志。
 * 必须在**原生事件**上读：React 的合成事件上这个字段有历史差异，
 * 且部分浏览器在 `keydown` 阶段只在 `nativeEvent` 上置位。
 *
 * 还有一种漏网情况：部分输入法（尤其 macOS 的部分候选模式）在组词结束后
 * 会补发一次 `keydown`，此时 `isComposing` 已经变 false 而候选词刚上屏。
 * 那一次回车是 IME 的收尾，不能再发。所以额外要求 `keyCode !== 229`
 * （229 是 IME 组词期间浏览器统一上报的 keyCode）。
 */
export function shouldSubmitOnEnter(e: {
  key: string;
  shiftKey: boolean;
  nativeEvent?: { isComposing?: boolean; keyCode?: number };
}): boolean {
  if (e.key !== "Enter") return false;
  if (e.shiftKey) return false; // Shift+Enter 是换行
  const ne = e.nativeEvent;
  // 组词中：这一下是选词，不是发送
  if (ne?.isComposing) return false;
  // IME 收尾补发的 keydown（keyCode 229）：候选词已上屏，但仍不是「发送」
  if (ne?.keyCode === 229) return false;
  return true;
}

export function ChatMessageTurn({ message, onRecall }: { message: Message; onRecall?: (messageId: number) => void }) {
  return (
    <div key={message.id} data-message-id={message.id} className={`message-turn message-turn--${message.role}`}>
      {message.role === "assistant" && message.activity && (
        <AgentActivity steps={message.activity.steps} activity={message.activity} streamText="" stale={false} error={null} status="done" showFinalAnswer={false} />
      )}
      <div className={`msg msg--${message.role}`}>
        {message.role === "assistant" ? (
          <div className="msg__bubble msg__bubble--md" dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }} />
        ) : (
          <div className="msg__bubble">{message.content}</div>
        )}
        {message.role === "user" && onRecall && (
          <button
            className="msg__recall"
            onClick={() => onRecall(message.id)}
            title="撤回这一问：删除它和后续回复，内容回到输入框"
          >撤回</button>
        )}
      </div>
    </div>
  );
}

export function ChatPanel({
  messages,
  onSend,
  onImportMedia,
  children,
  activeProjectId,
  task,
  taskError,
  stale,
  notice,
  onDismissNotice,
  onRefresh,
  steps = [],
  streamText = "",
  submitting = false,
  onStop,
  onRecall,
  draft,
  onDraftChange,
  modelName, modelSettingsStatus = "ready", onModelSettings,
  visible = true,
  timelineManaged = false,
}: {
  visible?: boolean;
  timelineManaged?: boolean;
  modelName?: string;
  modelSettingsStatus?: "loading" | "ready" | "error";
  onModelSettings?: () => void;
  messages: Message[];
  videos: Video[];
  onSend: (text: string) => void | boolean | Promise<boolean | void>;
  onImportMedia?: () => void;
  children?: React.ReactNode;
  onProcess?: (videoId: number) => void;
  onPlayVideo?: (videoId: number) => void;
  activeProjectId: number | null;
  task: TaskEvent | null;
  taskError: string | null;
  stale: boolean;
  notice: string | null;
  onDismissNotice: () => void;
  onRefresh: () => void;
  /** agent 活动流水（来自 SSE 的 step 事件）。 */
  steps: AgentStep[];
  /** 流式正文（来自 SSE 的 token 事件）。 */
  streamText: string;
  /**
   * 已按下发送、还在等后端返回 task_id。
   *
   * 实测这段 await 要 2245ms（服务端落库 + 起线程 + 返回 task_id）。
   * 没有它的话，这 2.2 秒里既没有 task 也没有 step，活动流整个不渲染，
   * 界面完全静止——用户投诉的「问个问题就不响应」就是这个窗口。
   *
   * 给默认值 false：这是个可选信号，没传就该当「没在提交」，
   * 不该让所有调用点都必须写。
   */
  submitting?: boolean;
  /**
   * 中止当前任务。没传就不显示停止按钮。
   *
   * 刻意做成可选而不是必填：ChatPanel 有 20 多个测试用例直接构造它，
   * 让每个都写一个假的 onStop 只会增加噪音。可选 + 缺省不渲染，
   * 行为与「不传就等于不支持中止」一致，不会有隐式副作用。
   */
  onStop?: () => void;
  /**
   * 撤回某条消息（连同其后的回复），并把原文填回输入框。
   * 同上，没传就不显示撤回按钮。
   */
  onRecall?: (messageId: number) => void;
  /**
   * 受控的消息输入框内容。**必须与 onDraftChange 一起传**，只传其一无效。
   *
   * 为什么需要把它提到 App 层：撤回要把原文填回输入框，而输入框的
   * state 原本在 ChatPanel 内部，App 拿不到也改不了。
   * 早先的绕法是给 ChatPanel 塞一个 ref + useEffect 监听计数器，
   * 那是把「一次赋值」伪装成「状态同步」，多一个中间层且更难推理 ——
   * 直接受控更简单。
   *
   * 两个都不传时退化为组件内部 state（20 多个既有测试用例都这么用，
   * 不该为了一个新功能让它们全体改造成受控）。
   */
  draft?: string;
  onDraftChange?: (text: string) => void;
}) {
  const modelSettingsLabel = modelSettingsStatus === "loading" ? "读取模型设置…"
    : modelSettingsStatus === "error" ? "设置读取失败" : "";
  const [localText, setLocalText] = useState("");
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState("");
  const pending = useRef(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const streamRef = useRef<HTMLDivElement>(null);

  // 受控优先。传了 draft 就以外部为准（撤回要把原文填回来），
  // 没传就用组件内部 state（既有测试与最小用法）。
  //
  // 判据是 `draft !== undefined` 而不是 `onDraftChange != null`：
  // 前者判「有没有受控」，后者判「有没有回调」，两者可能不一致
  // （只传 onDraftChange 会让输入框变成只读——比不受控更难排查）。
  const controlled = draft !== undefined;
  const text = controlled ? (draft as string) : localText;
  const setText = (next: string) => {
    if (controlled) onDraftChange?.(next);
    else setLocalText(next);
  };

  const current = useRef({text, activeProjectId});
  current.current = {text, activeProjectId};
  const busy = sending || submitting || task?.status === "pending" || task?.status === "running";
  const activeTask = submitting || task?.status === "pending" || task?.status === "running";
  const finalStreamText = task?.status === "cancelled" || task?.status === "failed" ? ""
    : task?.activity ? (streamText ? task.activity.segments.filter(s => s.phase === "final").map(s => s.text).join("\n") : "")
    : streamText;
  const timelineStreamText = timelineManaged && task?.status === "done" && messages[messages.length - 1]?.role === "assistant"
    ? ""
    : finalStreamText;
  const lastMessage = messages[messages.length - 1];
  const savedAssistantActivityShown = task?.status === "done"
    && lastMessage?.role === "assistant"
    && Boolean(lastMessage.activity);
  const timelineActivitySteps = steps.length > 0 ? steps : task?.activity?.steps ?? [];
  const showTimelineLiveActivity = timelineManaged && Boolean(task) && !savedAssistantActivityShown && (
    activeTask
    || task?.status === "cancelled"
    || task?.status === "failed"
    || timelineActivitySteps.length > 0
    || Boolean(task?.activity?.segments.some(segment => segment.phase !== "final"))
  );
  const submit = () => {
    if (!text.trim() || pending.current || busy) return;
    const sent = text;
    const project = activeProjectId;
    pending.current = true; setSending(true); setSendError("");
    const finish = (ok: boolean | void) => {
      if (ok !== false && current.current.activeProjectId === project && current.current.text === sent) setText("");
    };
    try {
      const result = onSend(sent);
      if (result instanceof Promise) {
        void result.then(finish).catch(() => setSendError("发送失败，内容已保留，请重试"))
          .finally(() => {pending.current = false; setSending(false)});
      } else {finish(result); pending.current = false; setSending(false)}
    } catch {setSendError("发送失败，内容已保留，请重试"); pending.current = false; setSending(false)}
  };
  useEffect(() => {
    if (textarea.current) {
      textarea.current.style.height = "auto";
      textarea.current.style.height = `${Math.min(240, Math.max(90, textarea.current.scrollHeight))}px`;
    }
  }, [text]);

  useEffect(() => {
    if (!visible) return;
    streamRef.current?.scrollTo({
      top: streamRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, task, taskError, steps, streamText, visible]);

  if (!activeProjectId) {
    return (
      <section className="chat-panel">
        <div className="empty-state">
          <div>选择一个项目，或新建一个</div>
          <div style={{ fontSize: 12 }}>
            左侧新建项目后，在这里粘贴 B站链接开始整理
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="chat-panel">
      <div className="chat-stream" ref={streamRef}><div className="chat-turns">
        {!timelineManaged && messages.length === 0 && <div className="conversation-welcome"><h2>今天想学什么？</h2><p>粘贴视频链接，或说说你想整理的内容。</p></div>}
        {!timelineManaged && messages.map(message => <ChatMessageTurn key={message.id} message={message} onRecall={onRecall} />)}

        {/* 一次性提示（错误等）。独立于消息流，刷新项目不会把它抹掉。 */}
        {notice && (
          <div className="hint-bar hint-bar--warn">
            <span style={{ whiteSpace: "pre-wrap" }}>{notice}</span>
            <button className="hint-bar__close" onClick={onDismissNotice}>
              知道了
            </button>
          </div>
        )}

        {!timelineManaged && !(task?.status==='done'&&messages[messages.length-1]?.role==='assistant'&&messages[messages.length-1]?.activity) && (activeTask||task?.status==='cancelled'||task?.status==='failed'||steps.length>0||finalStreamText) && <>
          <AgentActivity steps={steps} activity={task?.activity} streamText="" stale={stale} error={null} statusMessage={task?.message} submitting={submitting} status={task?.status} active={activeTask} showFinalAnswer={false}/>
          {finalStreamText&&<div className="msg msg--assistant"><div className="msg__bubble msg__bubble--md" dangerouslySetInnerHTML={{__html:renderMarkdown(finalStreamText)}}/></div>}
        </>}

        {children}

        {timelineManaged && submitting && !task && (
          <AgentActivity steps={[]} streamText="" stale={false} error={null} submitting status="pending" showFinalAnswer={false} />
        )}
        {showTimelineLiveActivity && task && (
          <AgentActivity
            steps={timelineActivitySteps}
            activity={task.activity}
            streamText=""
            stale={stale}
            error={null}
            statusMessage={task.message}
            status={task.status}
            active={activeTask}
            showFinalAnswer={false}
          />
        )}
        {timelineManaged && timelineStreamText && (
          <div className="msg msg--assistant"><div className="msg__bubble msg__bubble--md" dangerouslySetInnerHTML={{ __html: renderMarkdown(timelineStreamText) }} /></div>
        )}

        {/* 完成后给一条明确落点，否则进度泡凭空消失，用户不知道去看右栏。
            只在「没有走 agent 对话」时给 —— 走对话时回复本身就在上面，
            再补一句「知识点已更新」是噪声。 */}
        {!timelineManaged && task?.status === "done" && task.kind === "pipeline" && !!task.result && typeof task.result === "object" && "doc" in task.result && steps.length === 0 && !streamText && (
          <div className="msg msg--assistant">
            <div className="msg__bubble" style={{ background: "#eef7ee" }}>
              ✓ 整理完成，知识点已更新到右侧面板
            </div>
          </div>
        )}

        {/* 失败提示 */}
        {taskError && (
          <div className="msg msg--assistant">
            <div className="msg__bubble" style={{ background: "#fff0f0" }}>
              ⚠ {taskError}
            </div>
          </div>
        )}

        {/* 降级：SSE 长时间无更新（且进度确实卡住），给一个真正能用的恢复动作 */}
        {stale && !taskError && (
          <div className="hint-bar hint-bar--warn">
            正在处理（当前阶段较慢，进度可能长时间不变）。若长时间没有进展，可刷新状态。
            <button className="hint-bar__close" onClick={onRefresh}>
              刷新状态
            </button>
          </div>
        )}
      </div>

      </div>
      <div className="composer-wrap">
        <div className="composer">
          <textarea ref={textarea}
            aria-label="消息与视频链接"
            rows={3}
            placeholder="粘贴一个或多个视频链接，也可以直接提问…"
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => {if (shouldSubmitOnEnter(e)) {e.preventDefault(); submit();}}}
          />
          <div className="composer-toolbar">
            {onImportMedia && <button className="composer-add" onClick={onImportMedia} aria-label="添加本地视频或音频" title="添加本地视频或音频">＋</button>}
            <span className="composer-hint">Shift + Enter 换行</span>
            {onModelSettings&&<details className="composer-model">
              <summary>{modelName||modelSettingsLabel||'配置模型'} <span>⌄</span></summary>
              <div className="composer-model__menu">
                <span>{modelName||modelSettingsLabel||'尚未配置模型'}</span>
                {modelName&&modelSettingsStatus==='error'&&<span role="status">设置读取失败 · 显示上次读取的模型</span>}
                <button onClick={e=>{e.currentTarget.closest('details')?.removeAttribute('open');onModelSettings()}}>配置模型…</button>
              </div>
            </details>}
            {onStop && (submitting || task?.status === "pending" || task?.status === "running") ?
              <button className="composer-send" onClick={onStop} title="停止这一轮" aria-label="停止"><span className="sr-only">停止</span>■</button> :
              <button className="composer-send" onClick={submit} disabled={!text.trim() || busy} aria-label="发送" title="发送（Enter）">↑</button>}
          </div>
        </div>
        {sendError && <p role="alert" className="form-error">{sendError}</p>}
      </div>
    </section>
  );
}
