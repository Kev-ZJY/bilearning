import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { ActivitySnapshot, AgentStep } from "../../api/types";

export function formatElapsed(ms: number): string {
  if (ms < 1000) return "";
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}分${String(seconds % 60).padStart(2, "0")}s`;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return "不到 1 秒";
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} 秒`;
  const total = Math.floor(ms / 1000);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  if (minutes < 60) return seconds ? `${minutes} 分 ${seconds} 秒` : `${minutes} 分`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 小时 ${minutes % 60} 分`;
}

const ACTOR_LABELS: Record<string, string> = {
  orchestrator: "主编排",
  author: "整理角色",
  reviewer: "审查角色",
};

const STATUS_LABELS: Record<string, string> = {
  pending: "等待开始",
  running: "进行中",
  retrying: "正在重试",
  done: "成功",
  failed: "失败",
  error: "失败",
  blocked: "受阻",
  cancelled: "已取消",
  interrupted: "已中断",
};

const PIPELINE_STAGE_LABELS: Record<string, string> = {
  download: "准备素材",
  asr: "转写音频",
  frames: "整理画面证据",
  draft: "整理知识",
  verify: "核对完整性与来源",
  illustrate: "检查配图与例子",
  review: "独立审查",
  revise: "复核并修订",
  render: "校验并整合文档",
};

type StepIconKind = "read" | "search" | "frame" | "agent" | "model" | "generic";

function legacyConversationRound(step: AgentStep): number | null {
  if (typeof step.model === "string" && step.model.trim()) {
    return null;
  }
  if (typeof step.label !== "string") return null;
  const match = step.label.trim().match(/^调用模型\s*[（(]\s*第\s*(\d+)\s*轮\s*[）)]$/);
  if (!match) return null;
  const round = Number(match[1]);
  return Number.isSafeInteger(round) && round > 0 ? round : null;
}

function displayStepLabel(step: AgentStep): string {
  if (step.step === "pipeline_stage" && step.stage && PIPELINE_STAGE_LABELS[step.stage]) {
    return PIPELINE_STAGE_LABELS[step.stage];
  }
  if (step.step === "reviewer_cycle") return "审查进度";
  const legacyRound = legacyConversationRound(step);
  if (legacyRound !== null) return `会话第${legacyRound}轮`;
  if (step.step === "llm_call") return "调用模型";
  if (step.tool_name === "read_video_evidence") return "读取视频原始证据";
  if (step.tool_name === "view_candidate_figure") return "查看知识配图";
  if (step.tool_name === "view_video_frame") {
    const args = step.arguments;
    const second = args && typeof args === "object" && !Array.isArray(args)
      ? (args as Record<string, unknown>).second
      : undefined;
    return typeof second === "number" && Number.isFinite(second) && second >= 0
      ? `查看视频画面 · ${second} 秒`
      : "查看视频画面";
  }
  const isEditorialWebTool = step.tool_name === "search_web" || step.tool_name === "read_web";
  if (step.actor === "reviewer" && isEditorialWebTool) {
    const action = step.tool_name === "search_web" ? "查阅核验资料" : "核对知识内容";
    const domain = videoReferenceDomain(step);
    return domain ? `${action} · ${domain}` : action;
  }
  if (step.actor === "author" && isEditorialWebTool) {
    const action = step.tool_name === "search_web" ? "查阅视频相关资料" : "核对视频内容";
    const domain = videoReferenceDomain(step);
    return domain ? `${action} · ${domain}` : action;
  }
  if (step.tool_name === "search_web") return "搜索外部依据";
  if (step.tool_name === "read_web") return "读取网页正文";
  return step.label || step.step;
}

function stepIconKind(step: AgentStep): StepIconKind {
  if (["read_web", "read_video_evidence"].includes(step.tool_name ?? "")) return "read";
  if (step.tool_name === "search_web") return "search";
  if (step.tool_name === "view_video_frame") return "frame";
  if (step.step === "llm_call") return "model";
  if (step.execution === "child_task" || step.execution === "isolated_context"
      || (step.actor && step.actor !== "orchestrator")) return "agent";
  return "generic";
}

function StepIcon({ kind }: { kind: StepIconKind }) {
  if (kind === "read") {
    return <svg className="trace__icon trace__icon--read" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M3 2.5h6l3 3v8H3zM9 2.5v3h3M5.5 8h4M5.5 10.5h4" />
    </svg>;
  }
  if (kind === "search") {
    return <svg className="trace__icon trace__icon--search" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="6.8" cy="6.8" r="4.3" /><path d="m10 10 3.5 3.5" />
    </svg>;
  }
  if (kind === "frame") {
    return <svg className="trace__icon trace__icon--frame" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="2.3" y="2.5" width="11.4" height="11" rx="1.4" />
      <circle cx="10.4" cy="5.4" r="1.1" /><path d="m3.5 11 3-3 2.1 2 1.3-1.2 2.6 2.5" />
    </svg>;
  }
  if (kind === "agent") {
    return <svg className="trace__icon trace__icon--agent" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="5.2" r="2.1" /><path d="M3.7 13.2c.3-2.2 1.8-3.5 4.3-3.5s4 1.3 4.3 3.5" />
    </svg>;
  }
  if (kind === "model") {
    return <svg className="trace__icon trace__icon--model" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="4" y="4" width="8" height="8" rx="1.2" />
      <path d="M6.5 1.8v2M9.5 1.8v2M6.5 12v2.2M9.5 12v2.2M1.8 6.5H4M1.8 9.5H4M12 6.5h2.2M12 9.5h2.2" />
    </svg>;
  }
  return <svg className="trace__icon trace__icon--generic" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <path d="M2.5 8h10M8.5 4l4 4-4 4" />
  </svg>;
}

function timestamp(value: number | string | null | undefined): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function formatTimestamp(value: number | string): string {
  const milliseconds = timestamp(value);
  if (milliseconds === null) return String(value);
  const date = new Date(milliseconds);
  return Number.isFinite(date.getTime()) ? date.toLocaleString() : String(value);
}

function stepDuration(step: AgentStep, now: number, frozen: boolean): number | null {
  const reported = typeof step.elapsed_ms === "number" && Number.isFinite(step.elapsed_ms)
    ? Math.max(0, step.elapsed_ms)
    : null;
  const start = timestamp(step.started_at);
  const finish = timestamp(step.finished_at);
  const hasFinishedAt = finish !== null;
  if (!frozen && !hasFinishedAt && ["running", "retrying"].includes(step.status)) {
    const sinceStart = start !== null && now >= start ? now - start : null;
    if (reported === null && sinceStart === null) return null;
    return Math.max(reported ?? 0, sinceStart ?? 0);
  }
  if (reported !== null && reported > 0) return reported;
  if (start !== null && finish !== null && finish >= start) return finish - start;
  if (reported !== null) return reported;
  return null;
}

function visibleStepStatus(step: AgentStep, historical: boolean): string {
  if (historical && ["running", "pending", "retrying"].includes(step.status)) return "interrupted";
  if (step.status === "error") return "failed";
  return step.status;
}

function formatCount(value: number | null | undefined): string | null {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat("zh-CN").format(value)
    : null;
}

function formatTraceDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const seconds = totalSeconds % 60;
  const minutes = Math.floor(totalSeconds / 60);
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m ${seconds}s`;
  }
  if (minutes > 0) return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  return `${totalSeconds}s`;
}

function stringifyPublicValue(value: unknown, limit = 4000): string {
  let rendered: string;
  if (typeof value === "string") rendered = value;
  else {
    try {
      rendered = JSON.stringify(value, null, 2) ?? String(value);
    } catch {
      rendered = "[无法显示此字段]";
    }
  }
  return rendered.length > limit ? `${rendered.slice(0, limit)}…（已截断）` : rendered;
}

function videoReferenceDomain(step: AgentStep): string | null {
  const args = step.arguments;
  const url = args && typeof args === "object" && !Array.isArray(args)
    ? (args as Record<string, unknown>).url
    : undefined;
  if (typeof url !== "string") return null;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
    const hostname = parsed.hostname.toLowerCase().replace(/^www\./, "");
    return hostname && hostname.length <= 160 ? hostname : null;
  } catch {
    return null;
  }
}

function StepField({ label, value }: { label: string; value: unknown }) {
  if (value === undefined || value === null || value === "") return null;
  return (
    <section className="step__field">
      <span className="step__field-label">{label}</span>
      <pre>{stringifyPublicValue(value)}</pre>
    </section>
  );
}

function ToolRow({ step, historical, now }: { step: AgentStep; historical: boolean; now: number }) {
  const status = visibleStepStatus(step, historical);
  const [expanded, setExpanded] = useState(false);
  const statusClass = ["pending", "running", "retrying", "done", "failed", "blocked", "cancelled", "interrupted"].includes(status)
    ? status
    : "pending";
  const duration = stepDuration(step, now, historical);
  const detailAvailable = Boolean(
    step.status || step.detail || step.arguments !== undefined || step.result_summary || step.error || step.persistence_warning || duration !== null
      || step.cycle != null || step.attempts != null || step.retry_count != null || step.finish_reason
      || step.prompt_tokens != null || step.completion_tokens != null || step.reasoning_tokens != null
      || step.total_tokens != null || step.max_tokens != null || step.http_status != null,
  );
  const label = displayStepLabel(step);
  const visibleStatus = status === "done" && step.step === "reviewer_round"
    ? "本轮输出已返回"
    : status === "done" && legacyConversationRound(step) !== null
      ? "本轮已返回"
      : STATUS_LABELS[status] ?? status;

  return (
    <li className={`step trace__tool-row step--${statusClass}`} data-step-seq={step.seq}>
      <button
        className="step__toggle trace__action"
        type="button"
        onClick={() => setExpanded(value => !value)}
        aria-expanded={expanded}
        aria-label={`${label}，${visibleStatus}`}
      >
        <StepIcon kind={stepIconKind(step)} />
        <span className="step__label">{label}</span>
        {status !== "done" && (
          <span className={`step__state trace__status step__state--${statusClass}`}>{visibleStatus}</span>
        )}
        {detailAvailable && <svg className={`trace__row-chevron${expanded ? " is-open" : ""}`} viewBox="0 0 12 12" fill="none" aria-hidden="true"><path d="m3 4.5 3 3 3-3" /></svg>}
      </button>
      {expanded && detailAvailable && (
        <div className="step__details trace__details">
          {step.detail && <p className="step__detail">{step.detail}</p>}
          <span className="step__meta">状态：{visibleStatus}</span>
          {duration !== null && <span className="step__meta">用时：{formatDuration(duration)}</span>}
          {step.agent_id && <span className="step__meta">agent_id: {step.agent_id}</span>}
          {step.parent_task_id && <span className="step__meta">父任务: {step.parent_task_id}</span>}
          {step.child_task_id && <span className="step__meta">子任务: {step.child_task_id}</span>}
          {step.stage && <span className="step__meta">阶段: {step.stage}</span>}
          {step.label && step.label !== label && <span className="step__meta">原始标签: {step.label}</span>}
          {step.actor && <span className="step__meta">角色: {ACTOR_LABELS[step.actor] ?? step.actor}</span>}
          {step.execution && <span className="step__meta">执行方式: {step.execution === "isolated_context" ? "隔离上下文" : step.execution === "child_task" ? "子任务" : "工作流"}</span>}
          {step.tool_name && <span className="step__meta">工具: {step.tool_name}</span>}
          {step.round != null && <span className="step__meta">第 {step.round} 轮</span>}
          {step.model && <span className="step__meta">模型: {step.model}</span>}
          {step.max_tokens != null && <span className="step__meta">本次输出上限：{formatCount(step.max_tokens)} tokens</span>}
          {step.cycle != null && <span className="step__meta">循环 {step.cycle}</span>}
          {step.attempts != null && <span className="step__meta">尝试 {step.attempts} 次</span>}
          {step.retry_count != null && <span className="step__meta">重试 {step.retry_count} 次</span>}
          {step.finish_reason && <span className="step__meta">finish_reason: {step.finish_reason}</span>}
          {[step.prompt_tokens, step.completion_tokens, step.reasoning_tokens, step.total_tokens].some(value => value != null) && (
            <span className="step__meta">
              Tokens：
              {step.prompt_tokens != null && ` prompt ${formatCount(step.prompt_tokens)}`}
              {step.completion_tokens != null && ` · completion ${formatCount(step.completion_tokens)}`}
              {step.reasoning_tokens != null && ` · reasoning ${formatCount(step.reasoning_tokens)}`}
              {step.total_tokens != null && ` · total ${formatCount(step.total_tokens)}`}
            </span>
          )}
          {step.http_status != null && <span className="step__meta">HTTP {step.http_status}</span>}
          {step.started_at && <span className="step__meta">开始：{formatTimestamp(step.started_at)}</span>}
          {step.finished_at && <span className="step__meta">结束：{formatTimestamp(step.finished_at)}</span>}
          <StepField label="参数" value={step.arguments} />
          <StepField label="结果摘要" value={step.result_summary?.trim() === step.detail?.trim() ? undefined : step.result_summary} />
          <StepField label="错误" value={step.error} />
          {step.persistence_warning && <p className="step__warning">{step.persistence_warning}</p>}
          {step.truncated && <p className="step__warning">后端已截断部分活动内容。</p>}
        </div>
      )}
    </li>
  );
}

type ActivityEntry =
  | { order: number; key: string; kind: "step"; step: AgentStep }
  | { order: number; key: string; kind: "summary"; text: string };

function getTerminalStatus({
  status,
  historical,
  submitting,
  done,
  active,
}: { status?: string; historical: boolean; submitting: boolean; done: boolean; active: boolean }): string | null {
  if (submitting) return null;
  if (historical && status && ["running", "pending", "retrying"].includes(status)) return "interrupted";
  if (status === "error") return "failed";
  if (status && ["done", "failed", "blocked", "cancelled", "interrupted"].includes(status)) return status;
  if (done && !active) return "done";
  return null;
}

export function AgentActivity({
  steps,
  streamText,
  stale,
  error,
  active = false,
  submitting = false,
  done = false,
  status,
  statusMessage,
  activity,
  historical = false,
  initiallyExpanded = false,
  showFinalAnswer = true,
  onExpandedChange,
  actions,
  children,
}: {
  steps: AgentStep[];
  streamText: string;
  stale: boolean;
  error: string | null;
  active?: boolean;
  submitting?: boolean;
  done?: boolean;
  status?: string;
  statusMessage?: string;
  activity?: ActivitySnapshot;
  historical?: boolean;
  initiallyExpanded?: boolean;
  showFinalAnswer?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  actions?: ReactNode;
  children?: ReactNode;
}) {
  const hasLiveWork = getTerminalStatus({ status, historical, submitting, done, active }) === null && !historical && (
    active || submitting || ["pending", "running", "retrying"].includes(status ?? "")
      || steps.some(step => ["pending", "running", "retrying"].includes(step.status))
  );
  const [expanded, setExpanded] = useState(initiallyExpanded || hasLiveWork);
  const userChangedExpanded = useRef(false);
  const mountedAt = useRef(Date.now());
  const detail = useRef<HTMLDivElement>(null);
  const onExpandedChangeRef = useRef(onExpandedChange);
  const [now, setNow] = useState(Date.now());
  const terminalStatus = getTerminalStatus({ status, historical, submitting, done, active });
  const finished = terminalStatus !== null;

  const previousLiveWork = useRef(hasLiveWork);
  useEffect(() => {
    if (!previousLiveWork.current && hasLiveWork && !userChangedExpanded.current) setExpanded(true);
    previousLiveWork.current = hasLiveWork;
  }, [hasLiveWork]);

  useEffect(() => {
    onExpandedChangeRef.current = onExpandedChange;
  }, [onExpandedChange]);

  useEffect(() => {
    onExpandedChangeRef.current?.(expanded);
  }, [expanded]);

  useEffect(() => {
    if ((!active && !submitting) || finished) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active, submitting, finished]);

  useEffect(() => {
    if (expanded) detail.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [expanded]);

  const segments = activity?.segments ?? [];
  const summaries = segments.filter(segment => segment.phase !== "final");
  const finalSegments = segments.filter(segment => segment.phase === "final");
  const finalText = showFinalAnswer
    ? finalSegments.map(segment => segment.text).join("\n") || streamText
    : "";
  const entries: ActivityEntry[] = [
    ...steps.map(step => ({
      order: step.order ?? step.seq,
      key: `step:${step.actor ?? ""}:${step.agent_id ?? ""}:${step.child_task_id ?? ""}:${step.seq}`,
      kind: "step" as const,
      step,
    })),
    ...summaries.map(segment => ({
      order: segment.order,
      key: `summary:${segment.id}`,
      kind: "summary" as const,
      text: segment.text,
    })),
  ].sort((a, b) => a.order - b.order);

  const stepDurations = steps.map(step => stepDuration(step, now, historical || finished));
  const knownStepDurations = stepDurations.map(value => value ?? 0);
  const stepDurationMax = Math.max(0, ...knownStepDurations);
  const hasActivityDuration = typeof activity?.elapsed_ms === "number" && Number.isFinite(activity.elapsed_ms);
  const activityDuration = hasActivityDuration ? Math.max(0, activity.elapsed_ms) : 0;
  const duration = activityDuration > 0
    ? activityDuration
    : stepDurationMax || (!finished && !historical ? Math.max(0, now - mountedAt.current) : 0);
  const durationKnown = hasActivityDuration || stepDurations.some(value => value !== null) || (!finished && !historical);

  if (!steps.length && !segments.length && !streamText && !error && !active && !submitting && !finished) return null;

  const durationText = durationKnown ? formatTraceDuration(duration) : null;
  const toggleLabel = durationText
    ? `用时 ${durationText}，${expanded ? "收起" : "展开"}工作记录`
    : `工作记录，${expanded ? "收起" : "展开"}工作记录`;

  return (
    <div className="agent-process trace">
      <div className="trace__header">
        <button
          className="trace__disclosure"
          type="button"
          onClick={() => {
            userChangedExpanded.current = true;
            setExpanded(value => !value);
          }}
          aria-expanded={expanded}
          aria-label={toggleLabel}
        >
          <span>{durationText ? `用时 ${durationText}` : "工作记录"}</span>
          <svg className={`trace__chevron${expanded ? " is-open" : ""}`} viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <path d="m3 4.5 3 3 3-3" />
          </svg>
        </button>
        {actions && <div className="trace__actions">{actions}</div>}
      </div>

      {expanded && (
        <div className="activity--detail trace__body" ref={detail}>
          {stale && <p className="trace__notice">连接不稳定</p>}
          {terminalStatus === "cancelled" && <p className="trace__terminal">已停止，尚未生成最终答案</p>}
          {terminalStatus === "failed" && <p className="trace__terminal">处理失败</p>}
          {terminalStatus === "blocked" && <p className="trace__terminal">任务受阻</p>}
          {terminalStatus === "interrupted" && <p className="trace__terminal">历史任务已中断</p>}
          {(!steps.length || (statusMessage && !steps.some(step => step.status === "running" || step.status === "retrying"))) && !finished && (
            <div className="activity__waiting trace__waiting">
              <svg className="trace__icon" viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="5.5" /><path d="M8 4.5V8l2.5 1.5" /></svg>
              <span>{submitting ? "正在提交" : statusMessage || "正在接入"}</span>
            </div>
          )}
          {entries.length > 0 && (
            <ol className="activity__steps trace__entries">
              {entries.map(entry => entry.kind === "step" ? (
                <ToolRow key={entry.key} step={entry.step} historical={historical || finished} now={now} />
              ) : (
                <li className="activity__commentary trace__summary" key={entry.key}>{entry.text}</li>
              ))}
            </ol>
          )}
          {finalText && (
            <section className="activity__final trace__final" aria-label="最终回答">
              <h4>最终回答{!finished && "（生成中）"}</h4>
              <div className="trace__final-text">{finalText}</div>
            </section>
          )}
          {error && <div className="activity__error trace__error" role="alert">{error}</div>}
          {children}
        </div>
      )}
    </div>
  );
}
