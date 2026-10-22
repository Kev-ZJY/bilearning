import { useEffect, useRef, useState } from 'react';
import * as api from '../../api/client';
import type { AgentStep, KnowledgeDoc, Message, ProjectTaskRecord, TaskActivityIdentity, TaskActivityResponse, TaskEvent } from '../../api/types';
import { AgentActivity } from './AgentActivity';
import { ChatMessageTurn } from '../chat/ChatPanel';
import { buildConversationTimeline, type ConversationTaskEntry } from './conversationTimeline';

const RECOVERY_ID_PATTERN = /^[a-f\d]{24}$/i;
const FAILURE_SUMMARY_LIMIT = 180;
const RESUME_SUBMIT_TIMEOUT_MS = 15_000;
const ACTIVITY_POLL_INTERVAL_MS = 2_000;
const RESUME_SUBMIT_TIMEOUT_MESSAGE = '继续整理状态暂未确认（等待超时），任务列表仍会自动刷新；请先核对任务列表后再决定是否重试。';
const RESUME_POLL_TIMEOUT_MESSAGE = '继续整理已提交，但任务状态暂未确认（等待超时）；任务列表仍会自动刷新，请先核对当前任务状态。';
const EXPORT_ARGUMENTS_DENYLIST = /(?:api.?key|access.?token|secret|password|authorization|cookie|credential|prompt|messages?|transcript|transcription|answer|completion|content|base.?url|endpoint|setting)/i;
const BUDGET_FIELD_PATTERN = /(?:token|budget|limit|remaining|used|round)/i;
const MAX_CONSECUTIVE_ACTIVITY_FAILURES = 15;
const SECRET_TEXT_PATTERNS = [
  /\bBearer\s+[A-Za-z0-9._~+/-]+=*/gi,
  /\bsk-(?:ant-)?[A-Za-z0-9_-]{8,}\b/gi,
  /((?:api|access)[_-]?(?:key|token)|secret|password|authorization)\s*[:=]\s*[^\s,;"'}]+/gi,
  /\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g,
];

function sanitizePublicText(value: string, limit = 4_000): string {
  let sanitized = value;
  for (const pattern of SECRET_TEXT_PATTERNS) {
    sanitized = sanitized.replace(pattern, (_match, prefix: string | undefined) =>
      prefix ? `${prefix}=[已脱敏]` : '[已脱敏凭据]'
    );
  }
  return sanitized.length > limit ? `${sanitized.slice(0, limit)}…（已截断）` : sanitized;
}

function sanitizeArguments(value: unknown, depth = 0): unknown {
  if (depth > 5) return '[嵌套过深]';
  if (typeof value === 'string') return sanitizePublicText(value, 800);
  if (typeof value === 'number' || typeof value === 'boolean' || value == null) return value;
  if (Array.isArray(value)) return value.slice(0, 30).map(item => sanitizeArguments(item, depth + 1));
  if (typeof value !== 'object') return undefined;
  const output: Record<string, unknown> = {};
  for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
    if (EXPORT_ARGUMENTS_DENYLIST.test(key)) continue;
    const safe = sanitizeArguments(nested, depth + 1);
    if (safe !== undefined) output[key] = safe;
  }
  return output;
}

function safeBudget(value: Record<string, unknown> | null | undefined): Record<string, number> {
  const result: Record<string, number> = {};
  const visit = (item: unknown, prefix = '', depth = 0) => {
    if (!item || typeof item !== 'object' || depth > 3) return;
    for (const [key, nested] of Object.entries(item as Record<string, unknown>)) {
      const name = prefix ? `${prefix}.${key}` : key;
      if (typeof nested === 'number' && Number.isFinite(nested) && BUDGET_FIELD_PATTERN.test(key)) {
        result[name] = nested;
      } else if (nested && typeof nested === 'object') visit(nested, name, depth + 1);
    }
  };
  visit(value);
  return result;
}

function taskActivityIdentity(task: ProjectTaskRecord): TaskActivityIdentity | null {
  if (typeof task.task_id === 'string' && task.task_id.trim()) return { task_id: task.task_id };
  if (typeof task.video_id === 'number' && Number.isInteger(task.video_id) && task.video_id > 0
    && typeof task.recovery_id === 'string' && RECOVERY_ID_PATTERN.test(task.recovery_id)) {
    return { video_id: task.video_id, recovery_id: task.recovery_id };
  }
  return null;
}

function taskRowKey(projectId: number, task: ProjectTaskRecord, index: number): string {
  if (typeof task.task_id === 'string' && task.task_id.trim()) return `${projectId}:task:${task.task_id}`;
  if (typeof task.video_id === 'number' && typeof task.recovery_id === 'string' && RECOVERY_ID_PATTERN.test(task.recovery_id)) {
    return `${projectId}:recovery:${task.video_id}:${task.recovery_id}`;
  }
  return `${projectId}:unidentified:${index}:${task.status}:${task.stage ?? ''}`;
}

function isLiveStatus(status: string): boolean {
  return status === 'pending' || status === 'running' || status === 'retrying';
}

function exportStep(step: NonNullable<TaskActivityResponse['activity']>['steps'][number]) {
  return {
    seq: step.seq,
    step: step.step,
    label: sanitizePublicText(step.label, 300),
    status: step.status,
    order: step.order,
    actor: step.actor,
    agent_id: step.agent_id,
    parent_task_id: step.parent_task_id,
    child_task_id: step.child_task_id,
    stage: step.stage,
    round: step.round,
    cycle: step.cycle,
    attempts: step.attempts,
    retry_count: step.retry_count,
    tool_name: step.tool_name,
    arguments: sanitizeArguments(step.arguments),
    detail: step.detail ? sanitizePublicText(step.detail, 500) : undefined,
    result_summary: step.result_summary ? sanitizePublicText(step.result_summary, 600) : undefined,
    error: step.error ? sanitizePublicText(step.error, 1_200) : undefined,
    model: step.model ? sanitizePublicText(step.model, 200) : undefined,
    finish_reason: step.finish_reason ? sanitizePublicText(step.finish_reason, 100) : undefined,
    prompt_tokens: step.prompt_tokens,
    completion_tokens: step.completion_tokens,
    reasoning_tokens: step.reasoning_tokens,
    total_tokens: step.total_tokens,
    max_tokens: step.max_tokens,
    http_status: step.http_status,
    started_at: step.started_at,
    finished_at: step.finished_at,
    elapsed_ms: step.elapsed_ms,
    execution: step.execution,
    persistence_warning: step.persistence_warning ? sanitizePublicText(step.persistence_warning, 500) : undefined,
    truncated: step.truncated,
  };
}

function downloadActivityDiagnostic(projectId: number, task: ProjectTaskRecord, response: TaskActivityResponse) {
  const safeActivity = response.activity ? {
    started_at: response.activity.started_at,
    elapsed_ms: response.activity.elapsed_ms,
    steps: response.activity.steps.map(exportStep),
    // The final answer is displayed separately and is intentionally excluded from diagnostics.
    segments: response.activity.segments
      .filter(segment => segment.phase !== 'final')
      .map(segment => ({ id: segment.id, phase: segment.phase, order: segment.order, text: sanitizePublicText(segment.text, 600) })),
  } : null;
  const diagnostic = {
    exported_at: new Date().toISOString(),
    project_id: projectId,
    task: {
      task_id: response.task_id ?? task.task_id ?? null,
      video_id: task.video_id,
      recovery_id: task.recovery_id,
      status: response.status,
      stage: response.stage ?? task.stage ?? null,
      message: response.message ? sanitizePublicText(response.message, 500) : null,
      error: response.error ? sanitizePublicText(response.error, 1_500) : null,
      activity_available: response.activity_available,
      activity_source: response.activity_source,
      activity_warning: response.activity_warning ? sanitizePublicText(response.activity_warning, 500) : null,
      activity_truncated: response.activity_truncated ?? false,
      model_budget: safeBudget(response.model_budget),
    },
    activity: safeActivity,
  };
  const blob = new Blob([JSON.stringify(diagnostic, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  const basename = response.task_id ?? `video-${task.video_id ?? 'unknown'}-recovery-${task.recovery_id ?? 'unknown'}`;
  anchor.href = url;
  anchor.download = `workflow-${basename.replace(/[^A-Za-z0-9._-]/g, '_')}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

type TaskPresentation = 'manager' | 'conversation';

function TaskActivityPanel({
  projectId,
  task,
  identity,
  presentation = 'manager',
  allowTaskActions = false,
  resumeAvailable = canResume(task),
  busy = false,
  onStopTask,
  onResumeTask,
  onDraftTask,
  initiallyExpanded = true,
  priorAttempts = [],
}: {
  projectId: number;
  task: ProjectTaskRecord;
  identity: TaskActivityIdentity;
  presentation?: TaskPresentation;
  allowTaskActions?: boolean;
  resumeAvailable?: boolean;
  busy?: boolean;
  onStopTask?: () => void;
  onResumeTask?: () => void;
  onDraftTask?: () => void;
  initiallyExpanded?: boolean;
  priorAttempts?: ProjectTaskRecord[];
}) {
  const [response, setResponse] = useState<TaskActivityResponse | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [activityExpanded, setActivityExpanded] = useState(initiallyExpanded);
  const [actionsOpen, setActionsOpen] = useState(false);
  const [autoRefreshPaused, setAutoRefreshPaused] = useState(false);
  const [autoRefreshPauseMessage, setAutoRefreshPauseMessage] = useState('');
  const activityIdentityKey = JSON.stringify([
    projectId,
    identity.task_id ?? null,
    identity.video_id ?? null,
    identity.recovery_id ?? null,
  ]);
  const observedTaskState = useRef({ identityKey: activityIdentityKey, status: task.status });
  const pendingFinalRefreshIdentity = useRef<string | null>(null);

  useEffect(() => {
    if (!activityExpanded) return;
    let mounted = true;
    let requestGeneration = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    let consecutiveFailures = 0;
    const canPollThisTask = typeof identity.task_id === 'string' && isLiveStatus(task.status);
    const scheduleRefresh = () => {
      if (!canPollThisTask) return;
      if (consecutiveFailures >= MAX_CONSECUTIVE_ACTIVITY_FAILURES) {
        setAutoRefreshPaused(true);
        setAutoRefreshPauseMessage('连续读取失败已达上限；自动刷新已暂停，重新展开工作流可再次读取。');
        return;
      }
      timer = setTimeout(() => { void load(); }, ACTIVITY_POLL_INTERVAL_MS);
    };

    const load = async () => {
      const generation = ++requestGeneration;
      controller?.abort();
      const requestController = new AbortController();
      controller = requestController;
      try {
        const next = await api.taskActivity(projectId, identity, { signal: requestController.signal });
        if (!mounted || requestController.signal.aborted || generation !== requestGeneration) return;
        if (!isLiveStatus(task.status) && pendingFinalRefreshIdentity.current === activityIdentityKey) {
          pendingFinalRefreshIdentity.current = null;
        }
        setResponse(next);
        setError('');
        setLoading(false);
        consecutiveFailures = 0;
        if (canPollThisTask && next.activity_source === 'live' && isLiveStatus(next.status)) {
          scheduleRefresh();
        }
      } catch (caught) {
        if (!mounted || requestController.signal.aborted || generation !== requestGeneration) return;
        const message = caught instanceof Error && caught.message.trim() ? caught.message : '工作流读取失败，请稍后重试';
        setError(sanitizePublicText(message, 500));
        setLoading(false);
        consecutiveFailures += 1;
        scheduleRefresh();
      }
    };

    setError('');
    setLoading(response === null);
    setAutoRefreshPaused(false);
    setAutoRefreshPauseMessage('');
    void load();
    return () => {
      mounted = false;
      requestGeneration += 1;
      if (timer) clearTimeout(timer);
      controller?.abort();
    };
  }, [projectId, identity.task_id, identity.video_id, identity.recovery_id, task.status, activityExpanded]);

  useEffect(() => {
    const previous = observedTaskState.current;
    observedTaskState.current = { identityKey: activityIdentityKey, status: task.status };
    if (previous.identityKey !== activityIdentityKey) {
      pendingFinalRefreshIdentity.current = null;
      return;
    }
    if (isLiveStatus(previous.status) && !isLiveStatus(task.status)) {
      pendingFinalRefreshIdentity.current = activityIdentityKey;
    }
    if (pendingFinalRefreshIdentity.current !== activityIdentityKey
      || activityExpanded
      || typeof identity.task_id !== 'string'
      || !identity.task_id) return;

    let mounted = true;
    const controller = new AbortController();
    void api.taskActivity(projectId, identity, { signal: controller.signal }).then(next => {
      if (!mounted || controller.signal.aborted) return;
      pendingFinalRefreshIdentity.current = null;
      setResponse(next);
      setError('');
      setLoading(false);
      setAutoRefreshPaused(false);
      setAutoRefreshPauseMessage('');
    }).catch(caught => {
      if (!mounted || controller.signal.aborted) return;
      const message = caught instanceof Error && caught.message.trim() ? caught.message : '工作流读取失败，请稍后重试';
      setError(sanitizePublicText(message, 500));
      setLoading(false);
    });
    return () => {
      mounted = false;
      controller.abort();
    };
  }, [projectId, activityIdentityKey, identity.task_id, identity.video_id, identity.recovery_id, task.status, activityExpanded]);

  const budget = response ? safeBudget(response.model_budget) : {};
  const isHistorical = task.task_id === null || response?.activity_source === 'persisted';
  const effectiveStatus = response && !isLiveStatus(task.status) ? task.status : response?.status;
  const taskOptions = presentation === 'conversation' ? (
    <div className="trace-actions">
      <button type="button" aria-label="任务选项" aria-expanded={actionsOpen} onClick={() => setActionsOpen(open => !open)}>···</button>
      {actionsOpen && <div className="trace-actions__menu" aria-label="任务选项">
        {allowTaskActions && typeof task.task_id === 'string' && isLiveStatus(task.status) && onStopTask && (
          <button type="button" onClick={() => { setActionsOpen(false); onStopTask(); }}>停止</button>
        )}
        {allowTaskActions && resumeAvailable && onResumeTask && (
          <button type="button" disabled={busy} aria-busy={busy} onClick={() => { setActionsOpen(false); onResumeTask(); }}>
            {busy ? '正在提交…' : task.status === 'done' ? '复用素材重新整理' : '继续整理'}
          </button>
        )}
        {allowTaskActions && task.draft_available && typeof task.task_id === 'string' && onDraftTask && (
          <button type="button" onClick={() => { setActionsOpen(false); onDraftTask(); }}>查看待校验草稿</button>
        )}
        {response && <button type="button" onClick={() => { setActionsOpen(false); downloadActivityDiagnostic(projectId, task, response); }}>下载诊断 JSON</button>}
      </div>}
    </div>
  ) : null;

  if (presentation === 'conversation') {
    return (
      <div className="trace-task-workflow" data-task-id={task.task_id ?? undefined}>
        {activityExpanded && loading && response === null && <p className="trace-task-state" role="status">正在读取工作记录…</p>}
        {error && <p className="trace-task-error" role="alert">{error}</p>}
        {autoRefreshPaused && <p className="trace-task-state">工作记录读取失败，自动刷新已暂停。</p>}
        {response?.activity_persistence_error && (
          <details className="trace-task-persistence-error">
            <summary>活动记录未保存</summary>
            <p>{sanitizePublicText(response.activity_persistence_error, 600)}</p>
          </details>
        )}
        {response?.activity_available && response.activity ? (
          <AgentActivity
            key={`${task.task_id ?? task.source_task_id ?? task.source_recovery_id ?? 'task'}:activity`}
            steps={response.activity.steps}
            streamText=""
            stale={false}
            error={response.error && response.error.trim() !== task.error?.trim() ? sanitizePublicText(response.error, 1_500) : null}
            active={!isHistorical && isLiveStatus(response.status)}
            done={effectiveStatus === 'done'}
            status={effectiveStatus}
            statusMessage={response.message ?? undefined}
            activity={response.activity}
            historical={isHistorical}
            initiallyExpanded={activityExpanded}
            showFinalAnswer={false}
            onExpandedChange={setActivityExpanded}
            actions={taskOptions}
          >{renderPreviousAttempts(projectId, priorAttempts)}</AgentActivity>
        ) : (
          <>
            <AgentActivity
              key={`${task.task_id ?? task.source_task_id ?? task.source_recovery_id ?? 'task'}:activity`}
              steps={[]}
              streamText=""
              stale={false}
              error={error || response?.error ? sanitizePublicText(error || response?.error || '', 1_500) : null}
              active={isLiveStatus(task.status)}
              done={task.status === 'done'}
              status={task.status}
              statusMessage={response?.activity_available ? (response.message ?? undefined) : undefined}
              historical={isHistorical}
              initiallyExpanded={activityExpanded}
              showFinalAnswer={false}
              onExpandedChange={setActivityExpanded}
              actions={taskOptions}
            >{renderPreviousAttempts(projectId, priorAttempts)}</AgentActivity>
            {response && !response.activity_available && !response.error && (
              <p className="trace-task-unavailable">{sanitizePublicText(response.message || '尚无工作记录。', 300)}</p>
            )}
          </>
        )}
      </div>
    );
  }

  return (
    <section className="task-activity" aria-label="工作流详情">
      <header className="task-activity__header">
        <div>
          <h3>工作流诊断</h3>
          {response && <span className={`task-activity__source task-activity__source--${response.activity_source}`}>
            {response.activity_source === 'live' ? '实时活动' : response.activity_source === 'persisted' ? '已保存活动' : '无活动记录'}
          </span>}
        </div>
        {response && <button type="button" onClick={() => downloadActivityDiagnostic(projectId, task, response)}>下载诊断 JSON</button>}
      </header>
      <p className="task-activity__boundary">仅显示公开工作摘要和操作记录；不会推测或展示私有思维链。隔离上下文表示工作流隔离，不代表独立操作系统进程。</p>
      {loading && <p className="task-activity__state" role="status">正在读取工作流…</p>}
      {error && <p className="task-activity__error" role="alert">{error}</p>}
      {autoRefreshPaused && <p className="task-activity__state">{autoRefreshPauseMessage}</p>}
      {response && (
        <>
          <div className="task-activity__meta">
            {response.stage && <span>阶段：{sanitizePublicText(response.stage, 100)}</span>}
            {response.message && <span>{sanitizePublicText(response.message, 500)}</span>}
            {Object.keys(budget).length > 0 && <span>模型预算：{Object.entries(budget).map(([key, value]) => `${key} ${value}`).join(' · ')}</span>}
          </div>
          {response.activity_warning && <p className="task-activity__warning">{sanitizePublicText(response.activity_warning, 500)}</p>}
          {response.activity_persistence_error && (
            <details className="task-activity__warning">
              <summary>活动记录未保存</summary>
              <pre>{sanitizePublicText(response.activity_persistence_error, 600)}</pre>
            </details>
          )}
          {response.activity_truncated && <p className="task-activity__warning">活动记录已截断，诊断内容可能不完整。</p>}
          {response.activity_available && response.activity ? (
            <AgentActivity
              steps={response.activity.steps}
              streamText=""
              stale={false}
              error={response.error ? sanitizePublicText(response.error, 1_500) : null}
              active={!isHistorical && isLiveStatus(response.status)}
              done={effectiveStatus === 'done'}
              status={effectiveStatus}
              statusMessage={response.message ?? undefined}
              activity={response.activity}
              historical={isHistorical}
              initiallyExpanded={initiallyExpanded}
              onExpandedChange={setActivityExpanded}
            />
          ) : (
            <p className="task-activity__state">未记录可展示的公开过程摘要。{response.message ? ` ${sanitizePublicText(response.message, 300)}` : ''}</p>
          )}
        </>
      )}
    </section>
  );
}

function taskStatusLabel(status: string): string {
  if (status === 'done') return '已完成';
  if (status === 'failed' || status === 'error') return '失败';
  if (status === 'blocked') return '受阻';
  if (status === 'retrying') return '正在重试';
  if (status === 'cancelled') return '已取消';
  return '正在处理';
}

function canResume(task: ProjectTaskRecord): task is ProjectTaskRecord & { video_id: number; recovery_id: string } {
  return task.kind === 'pipeline'
    && (((task.status === 'failed' || task.status === 'cancelled') && task.resumable === true)
        || (task.status === 'done' && task.rebuildable === true))
    && typeof task.video_id === 'number'
    && Number.isInteger(task.video_id)
    && task.video_id > 0
    && typeof task.recovery_id === 'string'
    && RECOVERY_ID_PATTERN.test(task.recovery_id);
}

function recoverySourceKey(projectId: number, task: { video_id: number; recovery_id: string }): string {
  return `${projectId}:${task.video_id}:${task.recovery_id}`;
}

function failureDetails(task: ProjectTaskRecord): { summary: string; details: string } {
  const message = sanitizePublicText(task.message?.trim() ?? '');
  const error = sanitizePublicText(task.error?.trim() ?? '');
  const isGeneric = (value: string) => /^(?:traceback|error|exception|task failed|processing failed|任务失败|处理失败|整理失败|后台任务失败|完整错误详情)[\s:：.!。]*$/i.test(value);
  const summarize = (candidate: string) => {
    if (!candidate) return '';
    const usefulLines = candidate
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line && !/^(?:Traceback \(most recent call last\):|File ["'].*line \d+|at\s+.+|During handling of the above exception|The above exception was the direct cause)/i.test(line));
    const meaningfulLines = usefulLines.filter(line => !isGeneric(line));
    const hasTracebackStack = /^\s*Traceback \(most recent call last\):/m.test(candidate);
    const usefulLine = hasTracebackStack
      ? meaningfulLines[meaningfulLines.length - 1]
      : meaningfulLines[0];
    return usefulLine ?? '';
  };
  const summary = truncate(summarize(error) || summarize(message) || '任务处理失败', FAILURE_SUMMARY_LIMIT);
  const details = message && error && message !== error
    ? `任务消息：\n${message}\n\n错误详情：\n${error}`
    : message || error;
  return { summary, details: details.trim() === summary.trim() ? '' : details };
}

function attemptTimestamp(value: string | null | undefined): number | null {
  if (typeof value !== 'string' || !/(?:Z|[+-]\d{2}:\d{2})$/i.test(value.trim())) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function attemptTimeLabel(value: string | null | undefined): string | null {
  const timestamp = attemptTimestamp(value);
  if (timestamp === null) return null;
  const date = new Date(timestamp);
  const year = date.getFullYear();
  const dateLabel = year === new Date().getFullYear()
    ? `${date.getMonth() + 1}月${date.getDate()}日`
    : `${year}年${date.getMonth() + 1}月${date.getDate()}日`;
  const timeLabel = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
  return `${dateLabel} ${timeLabel}`;
}

function attemptDurationLabel(value: number | null | undefined): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return null;
  const milliseconds = Math.round(value);
  if (milliseconds < 1_000) return `${milliseconds}毫秒`;
  const totalSeconds = Math.floor(milliseconds / 1_000);
  const hours = Math.floor(totalSeconds / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}小时${minutes}分${seconds}秒`;
  if (minutes > 0) return `${minutes}分${seconds}秒`;
  return `${seconds}秒`;
}

function previousAttemptSummary(attempt: ProjectTaskRecord): string {
  return [
    attemptTimeLabel(attempt.created_at),
    taskStatusLabel(attempt.status),
    attemptDurationLabel(attempt.activity?.elapsed_ms),
  ].filter((part): part is string => Boolean(part)).join(' · ');
}

function previousAttemptDetails(attempt: ProjectTaskRecord): string {
  const { details } = failureDetails(attempt);
  const message = sanitizePublicText(attempt.message?.trim() ?? '');
  const error = sanitizePublicText(attempt.error?.trim() ?? '');
  const failed = attempt.status === 'failed' || attempt.status === 'error';
  if (!error && !failed) return '';
  return details || error || message;
}

function mostRecentAttemptsFirst(attempts: ProjectTaskRecord[]): ProjectTaskRecord[] {
  return attempts
    .map((attempt, index) => ({ attempt, index, time: attemptTimestamp(attempt.created_at) }))
    .sort((a, b) => {
      if (a.time !== null && b.time !== null && a.time !== b.time) return b.time - a.time;
      if (a.time !== null && b.time === null) return -1;
      if (a.time === null && b.time !== null) return 1;
      return a.index - b.index;
    })
    .map(item => item.attempt);
}

function PreviousAttemptRecord({ projectId, attempt, index }: {
  projectId: number;
  attempt: ProjectTaskRecord;
  index: number;
}) {
  const [activityOpen, setActivityOpen] = useState(false);
  const identity = taskActivityIdentity(attempt);
  const details = previousAttemptDetails(attempt);
  const attemptKey = attempt.task_id ?? attempt.source_task_id ?? attempt.recovery_id ?? `${attempt.created_at ?? 'unknown'}:${index}`;
  return (
    <details key={attemptKey}>
      <summary>{previousAttemptSummary(attempt)}</summary>
      {details && <pre>{details}</pre>}
      {identity && (
        <details onToggle={event => setActivityOpen(event.currentTarget.open)}>
          <summary>查看该尝试的工作记录</summary>
          {activityOpen && (
            <TaskActivityPanel
              key={`prior:${attemptKey}`}
              projectId={projectId}
              task={attempt}
              identity={identity}
              presentation="conversation"
              initiallyExpanded
            />
          )}
        </details>
      )}
    </details>
  );
}

function renderPreviousAttempts(projectId: number, attempts: ProjectTaskRecord[]) {
  if (attempts.length === 0) return null;
  const orderedAttempts = mostRecentAttemptsFirst(attempts);
  return (
    <details className="trace-task-attempts" aria-label="此前的继续整理尝试">
      <summary>此前 {attempts.length} 次尝试</summary>
      <div className="trace-task-attempts__items">
        {orderedAttempts.map((attempt, index) => <PreviousAttemptRecord key={attempt.task_id ?? attempt.source_task_id ?? attempt.recovery_id ?? `${attempt.created_at ?? 'unknown'}:${index}`} projectId={projectId} attempt={attempt} index={index} />)}
      </div>
    </details>
  );
}

function truncate(value: string, limit: number): string {
  const chars = Array.from(value);
  return chars.length > limit ? `${chars.slice(0, limit).join('')}…` : value;
}

type ProjectSnapshot = { generation: number; tasks: ProjectTaskRecord[] };
type ProjectNotice = { generation: number; message: string };
type BusySnapshot = { generation: number; taskKeys: Set<string> };

export function ProjectTasks({ projectId, onRefresh, onDraft, presentation = 'manager', representedTaskIds = [], messages = [], videosPresent = false, onRecall, liveTask = null, liveSteps = [] }: {
  projectId: number | null;
  onRefresh: () => void;
  onDraft: (doc: KnowledgeDoc | null) => void;
  presentation?: TaskPresentation;
  representedTaskIds?: string[];
  messages?: Message[];
  videosPresent?: boolean;
  onRecall?: (messageId: number) => void;
  liveTask?: TaskEvent | null;
  liveSteps?: AgentStep[];
}) {
  const [snapshot, setSnapshot] = useState<ProjectSnapshot>({ generation: 0, tasks: [] });
  const [busySnapshot, setBusySnapshot] = useState<BusySnapshot>({ generation: 0, taskKeys: new Set() });
  const [expandedWorkflowKey, setExpandedWorkflowKey] = useState<string | null>(null);
  const [consumedRecoverySources, setConsumedRecoverySources] = useState<Set<string>>(() => new Set());
  const [pollNotice, setPollNotice] = useState<ProjectNotice>({ generation: 0, message: '' });
  const [actionNotice, setActionNotice] = useState<ProjectNotice>({ generation: 0, message: '' });
  const actionNoticeOwner = useRef<symbol | null>(null);
  const currentProject = useRef<{ projectId: number | null; generation: number }>({ projectId: null, generation: 0 });
  const mounted = useRef(false);
  const activeResumeRequests = useRef(new Map<string, symbol>());
  const latestResumeTokens = useRef(new Map<string, symbol>());
  const submittedResumeTaskIds = useRef(new Map<number, Set<string>>());
  const confirmedResume = useRef<{ generation: number; taskId: string; token: symbol; observed: boolean } | null>(null);
  const refreshCallback = useRef(onRefresh);
  const draftCallback = useRef(onDraft);
  const pollNow = useRef<((pid: number, generation: number) => Promise<boolean>) | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  if (currentProject.current.projectId !== projectId) {
    currentProject.current = {
      projectId,
      generation: currentProject.current.generation + 1,
    };
  }
  const generation = currentProject.current.generation;
  refreshCallback.current = onRefresh;
  draftCallback.current = onDraft;

  const tasks = snapshot.generation === generation ? snapshot.tasks : [];
  const busyTaskKeys = busySnapshot.generation === generation ? busySnapshot.taskKeys : new Set<string>();
  const visiblePollError = pollNotice.generation === generation ? pollNotice.message : '';
  const visibleActionError = actionNotice.generation === generation ? actionNotice.message : '';
  const isCurrent = (pid: number, requestGeneration: number) =>
    mounted.current
    && currentProject.current.projectId === pid
    && currentProject.current.generation === requestGeneration;

  useEffect(() => {
    const pid = projectId;
    const effectGeneration = currentProject.current.generation;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let activePollController: AbortController | undefined;
    let requestSequence = 0;
    const seen = new Map<string, string>();
    const isLive = () => live && pid != null && isCurrent(pid, effectGeneration);

    setSnapshot({ generation: effectGeneration, tasks: [] });
    setPollNotice({ generation: effectGeneration, message: '' });
    actionNoticeOwner.current = null;
    confirmedResume.current = null;
    setActionNotice({ generation: effectGeneration, message: '' });
    setBusySnapshot({ generation: effectGeneration, taskKeys: new Set() });
    setExpandedWorkflowKey(null);
    if (pid == null) return () => { live = false; };

    const update = async (sequence = ++requestSequence) => {
      let changed = false;
      activePollController?.abort();
      const pollController = new AbortController();
      activePollController = pollController;
      try {
        const next = await api.projectTasks(pid, { signal: pollController.signal });
        if (!isLive() || sequence !== requestSequence) return false;
        const submittedIds = submittedResumeTaskIds.current.get(effectGeneration);
        next.forEach((task, index) => {
          const rowKey = taskRowKey(pid, task, index);
          const taskId = typeof task.task_id === 'string' ? task.task_id : null;
          if (seen.has(rowKey) && seen.get(rowKey) !== task.status) changed = true;
          if (taskId && submittedIds?.has(taskId) && ['done', 'failed', 'cancelled'].includes(task.status)) {
            changed = true;
            submittedIds.delete(taskId);
            const confirmed = confirmedResume.current;
            if (confirmed?.generation === effectGeneration && confirmed.taskId === taskId) {
              confirmed.observed = true;
              setActionNotice(previous => previous.generation === effectGeneration
                && previous.message === RESUME_POLL_TIMEOUT_MESSAGE
                ? { generation: effectGeneration, message: '' } : previous);
            }
          }
          seen.set(rowKey, task.status);
        });
        if (submittedIds && submittedIds.size === 0) submittedResumeTaskIds.current.delete(effectGeneration);
        setSnapshot({ generation: effectGeneration, tasks: next });
        setPollNotice({ generation: effectGeneration, message: '' });
        if (changed && isLive()) refreshCallback.current();
      } catch {
        if (isLive() && sequence === requestSequence) {
          setPollNotice({ generation: effectGeneration, message: '任务状态暂时不可用，将自动重试' });
        }
      } finally {
        if (activePollController === pollController) activePollController = undefined;
      }
      if (isLive() && sequence === requestSequence) {
        timer = setTimeout(() => {
          timer = undefined;
          void update();
        }, 2000);
      }
      return changed;
    };

    const refreshNow = async (requestedPid: number, requestedGeneration: number) => {
      if (requestedPid !== pid || requestedGeneration !== effectGeneration || !isLive()) return false;
      if (timer) clearTimeout(timer);
      timer = undefined;
      return await update(++requestSequence);
    };
    pollNow.current = refreshNow;
    void update();

    return () => {
      live = false;
      if (timer) clearTimeout(timer);
      activePollController?.abort();
      submittedResumeTaskIds.current.delete(effectGeneration);
      if (pollNow.current === refreshNow) pollNow.current = null;
    };
  }, [projectId, generation]);

  const resume = async (task: ProjectTaskRecord & { video_id: number; recovery_id: string }, rowKey: string) => {
    if (projectId == null) return;
    const pid = projectId;
    const requestGeneration = generation;
    if (!isCurrent(pid, requestGeneration)) return;
    const requestKey = `${requestGeneration}:${rowKey}`;
    if (activeResumeRequests.current.has(requestKey)) return;
    const requestToken = Symbol(requestKey);
    activeResumeRequests.current.set(requestKey, requestToken);
    latestResumeTokens.current.set(requestKey, requestToken);
    actionNoticeOwner.current = requestToken;
    confirmedResume.current = null;
    const clearBusy = () => setBusySnapshot(previous => {
      if (previous.generation !== requestGeneration) return previous;
      const taskKeys = new Set(previous.taskKeys);
      taskKeys.delete(rowKey);
      return { generation: requestGeneration, taskKeys };
    });
    setBusySnapshot(previous => {
      const taskKeys = previous.generation === requestGeneration ? new Set(previous.taskKeys) : new Set<string>();
      taskKeys.add(rowKey);
      return { generation: requestGeneration, taskKeys };
    });
    setActionNotice({ generation: requestGeneration, message: '' });
    let submissionAccepted = false;

    // A delayed POST or status refresh must not leave the source row stuck in
    // "submitting" forever. The timer only bounds frontend confirmation; it does
    // not cancel or change the backend task. Polling continues to show the task
    // status and any failure details.
    const submitTimeout = setTimeout(() => {
      if (confirmedResume.current?.token === requestToken && confirmedResume.current.observed) return;
      if (latestResumeTokens.current.get(requestKey) !== requestToken
        || actionNoticeOwner.current !== requestToken) return;
      if (activeResumeRequests.current.get(requestKey) === requestToken) {
        activeResumeRequests.current.delete(requestKey);
        if (isCurrent(pid, requestGeneration)) clearBusy();
      }
      if (!isCurrent(pid, requestGeneration)) {
        latestResumeTokens.current.delete(requestKey);
        return;
      }
      setActionNotice({
        generation: requestGeneration,
        message: submissionAccepted ? RESUME_POLL_TIMEOUT_MESSAGE : RESUME_SUBMIT_TIMEOUT_MESSAGE,
      });
    }, RESUME_SUBMIT_TIMEOUT_MS);

    try {
      const resumed = await (task.status === 'done' ? api.rebuildTask : api.resumeTask)(pid, task.video_id, task.recovery_id);
      submissionAccepted = true;
      setConsumedRecoverySources(previous => {
        const next = new Set(previous);
        next.add(recoverySourceKey(pid, task));
        return next;
      });
      if (!isCurrent(pid, requestGeneration)) return;
      if (activeResumeRequests.current.get(requestKey) === requestToken) {
        activeResumeRequests.current.delete(requestKey);
        clearBusy();
      }
      if (typeof resumed.task_id === 'string' && resumed.task_id) {
        confirmedResume.current = { generation: requestGeneration, taskId: resumed.task_id, token: requestToken, observed: false };
        const taskIds = submittedResumeTaskIds.current.get(requestGeneration) ?? new Set<string>();
        taskIds.add(resumed.task_id);
        submittedResumeTaskIds.current.set(requestGeneration, taskIds);
      }
      const refreshedByPoll = await pollNow.current?.(pid, requestGeneration);
      if (isCurrent(pid, requestGeneration)
        && latestResumeTokens.current.get(requestKey) === requestToken
        && !refreshedByPoll) refreshCallback.current();
    } catch (error) {
      if (isCurrent(pid, requestGeneration)
        && latestResumeTokens.current.get(requestKey) === requestToken
        && actionNoticeOwner.current === requestToken) {
        const detail = error instanceof Error && error.message.trim() ? error.message.trim() : '请稍后重试';
        setActionNotice({ generation: requestGeneration, message: `继续整理失败：${truncate(detail, FAILURE_SUMMARY_LIMIT)}` });
      }
    } finally {
      clearTimeout(submitTimeout);
      const ownsRequest = activeResumeRequests.current.get(requestKey) === requestToken;
      if (ownsRequest) {
        activeResumeRequests.current.delete(requestKey);
        if (isCurrent(pid, requestGeneration)) clearBusy();
      }
      if (latestResumeTokens.current.get(requestKey) === requestToken) latestResumeTokens.current.delete(requestKey);
      if (actionNoticeOwner.current === requestToken) actionNoticeOwner.current = null;
    }
  };

  const viewDraft = async (task: ProjectTaskRecord) => {
    if (projectId == null) return;
    if (typeof task.task_id !== 'string' || !task.task_id) return;
    const pid = projectId;
    const requestGeneration = generation;
    try {
      const doc = await api.taskDraft(pid, task.task_id);
      if (isCurrent(pid, requestGeneration)) draftCallback.current(doc);
    } catch {
      if (isCurrent(pid, requestGeneration)) {
        actionNoticeOwner.current = null;
        setActionNotice({ generation: requestGeneration, message: '草稿读取失败，请稍后重试' });
      }
    }
  };

  if (!projectId || (presentation === 'manager' && !tasks.length && !visiblePollError && !visibleActionError)) return null;

  const renderTaskRow = (task: ProjectTaskRecord, index: number) => {
    const taskIndex = tasks.indexOf(task);
    const rowKey = taskRowKey(projectId, task, taskIndex >= 0 ? taskIndex : index);
    const workflowIdentity = taskActivityIdentity(task);
    const workflowKey = workflowIdentity ? `${rowKey}:activity` : null;
    const hasRecoverySource = canResume(task);
    const sourceAlreadyConsumed = hasRecoverySource
      && consumedRecoverySources.has(recoverySourceKey(projectId, task));
    const recoverable = hasRecoverySource && !sourceAlreadyConsumed;
    const { summary, details } = task.status === 'failed' || task.status === 'error'
      ? failureDetails(task)
      : { summary: task.message?.trim() || task.error?.trim() || '', details: '' };
    const nonResumableReason = (task.status === 'failed' || task.status === 'error') && (
      sourceAlreadyConsumed
        ? '此恢复来源已用于创建新的继续整理任务，请查看最新任务记录。'
        : task.resumable === false && task.recovery_reason?.trim()
    );
    const isBusy = busyTaskKeys.has(rowKey);
    const workflowExpanded = workflowKey !== null && expandedWorkflowKey === workflowKey;
    const stopTask = () => {
      if (typeof task.task_id === 'string' && isLiveStatus(task.status)) void api.cancelTask(task.task_id);
    };

    return (
      <div className={`project-task${presentation === 'conversation' ? ' trace-history__row' : ''}`} key={rowKey} data-status={task.status}>
        <div className="project-task__content">
          <div className="project-task__heading">
            <strong>{task.kind === 'pipeline' ? '视频整理' : '助手任务'}</strong>
            <span className="project-task__status">{taskStatusLabel(task.status)}</span>
          </div>
          {summary && <span className="project-task__message">{summary}</span>}
          {recoverable && <p className="project-task__recovery-help">复用已保存的转写和画面分析，重新核对来源后继续整理。</p>}
          {nonResumableReason && <p className="project-task__recovery-reason">无法继续：{truncate(nonResumableReason, FAILURE_SUMMARY_LIMIT)}</p>}
          {details && (
            <details className="project-task__failure-details">
              <summary>查看完整错误详情</summary>
              <pre>{details}</pre>
            </details>
          )}
        </div>
        <div className="project-task__controls">
          {!['done', 'failed', 'error', 'blocked', 'cancelled'].includes(task.status)
            && typeof task.task_id === 'string'
            && <button onClick={stopTask}>停止</button>}
          {task.draft_available && typeof task.task_id === 'string'
            && <button onClick={() => { void viewDraft(task); }}>查看待校验草稿</button>}
          {workflowIdentity && workflowKey && (
            <button
              type="button"
              aria-expanded={workflowExpanded}
              onClick={() => setExpandedWorkflowKey(previous => previous === workflowKey ? null : workflowKey)}
            >
              {workflowExpanded ? '收起工作流' : '查看工作流'}
            </button>
          )}
          {recoverable && (
            <button
              className="project-task__resume"
              disabled={isBusy}
              aria-busy={isBusy}
              onClick={() => { void resume(task, rowKey); }}
            >
              {isBusy ? '正在提交…' : task.status === 'done' ? '复用素材重新整理' : '继续整理'}
            </button>
          )}
        </div>
        {workflowExpanded && workflowIdentity && (
          <TaskActivityPanel
            key={workflowKey}
            projectId={projectId}
            task={task}
            identity={workflowIdentity}
            presentation={presentation}
            resumeAvailable={recoverable}
            busy={isBusy}
            onStopTask={stopTask}
            onResumeTask={() => { if (recoverable) void resume(task, rowKey); }}
            onDraftTask={() => { void viewDraft(task); }}
          />
        )}
      </div>
    );
  };

  if (presentation === 'conversation') {
    const liveRecord: ProjectTaskRecord | null = liveTask?.task_id
      ? { ...liveTask, task_id: liveTask.task_id }
      : null;
    const conversationTasks = liveRecord ? [...tasks, liveRecord] : tasks;
    const conversationRepresented = [
      ...representedTaskIds,
      ...liveSteps.map(step => step.child_task_id).filter((id): id is string => typeof id === 'string' && id.length > 0),
    ];
    const timeline = buildConversationTimeline(messages, conversationTasks, conversationRepresented);
    const taskEntries = timeline.filter((entry): entry is ConversationTaskEntry => entry.kind === 'task_group');
    const taskEntryIndex = new Map(taskEntries.map((entry, index) => [entry.key, index]));

    const renderConversationTask = (entry: ConversationTaskEntry) => {
      const task = entry.latest;
      const taskIndex = taskEntryIndex.get(entry.key) ?? 0;
      const rowKey = taskRowKey(projectId, task, taskIndex);
      const identity = taskActivityIdentity(task);
      const previousAttempts = entry.attempts.filter(attempt => attempt !== entry.latest);
      const { summary, details } = task.status === 'failed' || task.status === 'error'
        ? failureDetails(task)
        : { summary: task.message?.trim() || task.error?.trim() || '', details: task.error?.trim() ?? '' };
      const recoverable = canResume(task)
        && !consumedRecoverySources.has(recoverySourceKey(projectId, task));
      const isBusy = busyTaskKeys.has(rowKey);
      const isLive = typeof task.task_id === 'string' && isLiveStatus(task.status);
      const stopTask = () => {
        if (typeof task.task_id === 'string' && isLiveStatus(task.status)) void api.cancelTask(task.task_id);
      };

      return (
        <article className="conversation-task-turn message-turn" key={entry.key} data-task-id={task.task_id ?? undefined} data-recovery-id={entry.source_recovery_id ?? undefined} data-status={task.status}>
          {identity && (
            <TaskActivityPanel
              key={`${entry.key}:${task.task_id ?? task.source_task_id ?? 'disk-only'}`}
              projectId={projectId}
              task={task}
              identity={identity}
              presentation="conversation"
              allowTaskActions
              resumeAvailable={Boolean(recoverable)}
              busy={isBusy}
              initiallyExpanded={isLive}
              priorAttempts={previousAttempts}
              onStopTask={stopTask}
              onResumeTask={() => { if (recoverable) void resume(task as ProjectTaskRecord & { video_id: number; recovery_id: string }, rowKey); }}
              onDraftTask={() => { void viewDraft(task); }}
            />
          )}
          <div className="project-task__content">
            <div className="project-task__heading">
              <strong>{task.kind === 'pipeline' ? '视频整理' : '助手任务'}</strong>
              <span className="project-task__status">{taskStatusLabel(task.status)}</span>
            </div>
            {summary && <span className="project-task__message">{summary}</span>}
            {task.status === 'failed' || task.status === 'error' ? (
              details && <details className="project-task__failure-details"><summary>查看完整错误详情</summary><pre>{details}</pre></details>
            ) : task.error ? (
              <details className="project-task__failure-details"><summary>查看完整错误详情</summary><pre>{sanitizePublicText(task.error, 2_000)}</pre></details>
            ) : null}
          </div>
        </article>
      );
    };

    return (
      <div className="conversation-timeline" aria-label="会话时间线">
        {messages.length === 0 && conversationTasks.length === 0 && !videosPresent && (
          <div className="conversation-welcome"><h2>今天想学什么？</h2><p>粘贴视频链接，或说说你想整理的内容。</p></div>
        )}
        {visiblePollError && <p role="status">{visiblePollError}</p>}
        {visibleActionError && <p className="project-tasks__action-error" role="alert">{visibleActionError}</p>}
        {timeline.map(entry => entry.kind === 'message'
          ? <ChatMessageTurn key={entry.key} message={entry.message} onRecall={onRecall} />
          : renderConversationTask(entry))}
      </div>
    );
  }

  return (
    <details className="project-tasks" aria-label="项目任务">
      <summary>任务 · {tasks.filter(task => !['done', 'failed', 'cancelled'].includes(task.status)).length} 个进行中</summary>
      {visiblePollError && <p role="status">{visiblePollError}</p>}
      {visibleActionError && <p className="project-tasks__action-error" role="alert">{visibleActionError}</p>}
      {tasks.slice(-6).reverse().map((task, index) => renderTaskRow(task, index))}
    </details>
  );
}
