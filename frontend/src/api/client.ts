import type { ProcessingOptions } from "./types";
import type {
  FocusRef,
  KnowledgeDoc,
  Message,
  Project,
  Settings,
  Video,
  Playback,
  ProjectTaskRecord,
  TaskActivityIdentity,
  TaskActivityResponse,
} from "./types";

// 前端用相对路径 /api/...，经 Vite proxy 转发到后端 127.0.0.1:8000，
// 浏览器走同源请求；后端另允许 5180 的开发源。

async function request<T>(
  path: string,
  opts: RequestInit = {},
  options: { hideErrorDetail?: boolean } = {},
): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    // 后端报错统一收敛为 400（如 b23.tv 短链无法解析），detail 是中文提示，
    // 需要透传给前端以助手消息形式展示，不要用 alert。
    let msg = `请求失败 (${res.status})`;
    try {
      const err = await res.json();
      if (!options.hideErrorDetail && err && typeof err.detail === "string") {
        msg = err.detail;
      }
    } catch {
      /* 非 JSON 错误体，保留默认文案 */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

const PROJECT_TASKS_TIMEOUT_MS = 8_000;

async function requestWithDeadline<T>(
  path: string,
  externalSignal?: AbortSignal,
  timeoutMessage = "任务列表读取超时，请稍后重试",
): Promise<T> {
  if (externalSignal?.aborted) {
    throw externalSignal.reason ?? new DOMException("The operation was aborted", "AbortError");
  }

  const controller = new AbortController();
  let rejectInterruption!: (reason: unknown) => void;
  const interrupted = new Promise<never>((_resolve, reject) => { rejectInterruption = reject; });
  const interrupt = (reason: unknown) => {
    if (!controller.signal.aborted) controller.abort(reason);
    rejectInterruption(reason);
  };
  const forwardAbort = () => interrupt(
    externalSignal?.reason ?? new DOMException("The operation was aborted", "AbortError")
  );
  externalSignal?.addEventListener("abort", forwardAbort, { once: true });
  if (externalSignal?.aborted) forwardAbort();

  const timeoutError = new Error(timeoutMessage);
  timeoutError.name = "TimeoutError";
  const timeout = setTimeout(() => interrupt(timeoutError), PROJECT_TASKS_TIMEOUT_MS);
  const requestPromise = (async () => {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
    });
    if (!res.ok) {
      let message = `请求失败 (${res.status})`;
      try {
        const err = await res.json();
        if (err && typeof err.detail === "string") message = err.detail;
      } catch {
        /* 非 JSON 错误体，保留默认文案 */
      }
      throw new Error(message);
    }
    return (await res.json()) as T;
  })();

  try {
    return await Promise.race([requestPromise, interrupted]);
  } finally {
    clearTimeout(timeout);
    externalSignal?.removeEventListener("abort", forwardAbort);
  }
}

export function listProjects(): Promise<Project[]> {
  return request<Project[]>("/api/projects");
}

export function createProject(body: {
  title: string;
  subject?: string;
  directory_grant?: string;
}): Promise<Project> {
  return request<Project>("/api/projects", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// 项目详情：包含 messages / knowledge。视频列表较精简（无 cover），
// 封面图片请走 listVideos。
export function getProject(pid: number): Promise<{
  id: number;
  title: string;
  subject?: string;
  storage_root?: string;
  existing_document?: boolean;
  storage_status?: {status:string;message:string};
  videos: Video[];
  messages: Message[];
  knowledge: KnowledgeDoc | null;
}> {
  return request(`/api/projects/${pid}`);
}

// 视频列表：字段最全（含 bvid / cid / cover）。聊天面板视频芯片用这个。
export function listVideos(pid: number): Promise<Video[]> {
  return request<Video[]>(`/api/projects/${pid}/videos`);
}

export function getPlayback(pid: number, videoId: number): Promise<Playback> {
  return request(`/api/projects/${pid}/videos/${videoId}/playback`);
}

export function addVideo(
  pid: number,
  body: { url: string; page_end?: number; reuse_existing?: boolean; title?: string; duration?: number; path?: string }
): Promise<Video & {videos?: Video[]}> {
  return request<Video>(`/api/projects/${pid}/videos`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function listMessages(pid: number): Promise<Message[]> {
  return request<Message[]>(`/api/projects/${pid}/messages`);
}

// 发消息 = 提交一个对话任务。后端是异步的（agent 工具循环要跑十几秒到几十秒），
// 立刻返回 {message, task_id}：message 是刚落库的用户消息（可乐观上屏），
// task_id 用来订阅 /api/tasks/{task_id}/events 看实时进度。
//
// 注意类型必须是 {message, task_id} 而不是 Message：写错时 tsc 会报
// TS2339（Property 'task_id' does not exist），而 vitest 照样全绿
// —— 泛型只约束编译期，运行时对象照样能取到字段。契约错配要靠 tsc 拦。
export function sendMessage(
  pid: number,
  body: { content: string; focus?: FocusRef | null; video_ids?: number[] }
): Promise<{ message: Message; task_id: string }> {
  return request<{ message: Message; task_id: string }>(
    `/api/projects/${pid}/messages`,
    {
      method: "POST",
      body: JSON.stringify(body),
    }
  );
}

// 提交处理任务，立即返回 task_id；进度走 SSE。
export function process(
  pid: number,
  body: { video_id: number; intent?: string; terms?: string[]; options?: ProcessingOptions }
): Promise<{ task_id: string }> {
  return request<{ task_id: string }>(`/api/projects/${pid}/process`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// 中止一个正在跑的任务。
//
// 返回 status 而不是恒返回 "cancelled"：后端把「停止一个已经完成的任务」
// 当作 no-op（用户慢 3 秒按停止很常见），此时 status 仍是 done。
// 前端据此知道**不要**把界面上已经显示的答案标成被取消。
//
// 404 不当作错误处理：任务刚好自己跑完时后端会返回 404，
// 而「点了停止、任务已结束」是完全正常的时序，不该弹报错。
export async function cancelTask(
  taskId: string
): Promise<{ ok: true; status: string } | null> {
  try {
    return await request<{ ok: true; status: string }>(
      `/api/tasks/${taskId}/cancel`,
      { method: "POST" }
    );
  } catch {
    // 任务不存在 / 后端不可达：停止这个动作已经生效或不适用，
    // 不该因此给用户一个红色报错。返回 null 表示「不用管」。
    return null;
  }
}

// 撤回一条消息：中止仍在跑的任务 + 删掉这一问连同它的回复。
//
// restored 是被删掉的用户消息原文 —— 前端要把它填回输入框，
// 这是「撤回」与「删除」的本质区别：用户改一个字就能重发，不用重打。
export function recallMessage(
  pid: number,
  messageId: number
): Promise<{ ok: true; restored: string; removed: number }> {
  return request<{ ok: true; restored: string; removed: number }>(
    `/api/projects/${pid}/messages/${messageId}`,
    { method: "DELETE" }
  );
}

// SSE 断流时的兜底：轮询等 assistant 回复落库。
//
// 为什么需要：对话的回复**本身就是产品**，但它只在 SSE 发 done 时才被刷新到
// 界面。SSE 一断（切项目、网络抖动、EventSource 被浏览器回收），用户就只
// 看到自己那条消息孤零零挂着，助手回复永远不出现——而回复其实早就在库里了。
// 视频流水线断流可以提示用户「去刷新」，对话不行：用户刚问完，最在意的
// 就是答案会不会来。
//
// 三条约束：
// 1. **有上限**：maxAttempts 次后返回 null，绝不无限打后端。
// 2. **只认新消息**：按 after_id 过滤，否则会把上一轮的回复当成本轮的，
//    界面上表现为「答案突然变成上次那个」。
// 3. **不抛异常**：网络错误直接返回 null，兜底路径不该再制造一个错误。
export async function pollForAssistantReply(
  pid: number,
  afterId: number,
  opts: { intervalMs?: number; maxAttempts?: number } = {}
): Promise<Message | null> {
  const intervalMs = opts.intervalMs ?? 1500;
  const maxAttempts = opts.maxAttempts ?? 40; // 默认 40 × 1.5s ≈ 1 分钟
  for (let i = 0; i < maxAttempts; i += 1) {
    if (i > 0) await new Promise((r) => setTimeout(r, intervalMs));
    try {
      const list = await listMessages(pid);
      const hit = list.find(
        (m) => m.role === "assistant" && m.id > afterId
      );
      if (hit) return hit;
    } catch {
      return null; // 后端不可达：兜底本身不该再抛一次
    }
  }
  return null;
}

// ---- 后端设置（LLM 配置入口）----
//
// 类型里**没有** llm_api_key 明文：后端从不把它发到前端，
// 前端也就没有机会不小心把它打进日志或截图里。见 Settings 类型注释。

export function getSettings(): Promise<Settings> {
  return request<Settings>("/api/settings");
}

// 局部更新。只传要改的字段；传 null 表示「这一项恢复成环境变量默认值」。
//
// 类型用 Record<string, unknown> 而不是 Partial<Settings>：
// 后端接受 null（= 清除覆盖），而 Partial<Settings> 的字段都是必填的
// 具体类型，null 过不了类型检查 —— 而「恢复默认」正是这个端点必须支持的
// 操作之一（界面上有「恢复默认」按钮）。
export function updateSettings(
  body: Record<string, unknown>
): Promise<Settings> {
  return request<Settings>("/api/settings", {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

// 本次运行配置只提交接口地址、模型和密钥。这个路由不更新持久化设置，
// 也不把服务端错误 detail 回传到 UI，以防错误正文意外包含请求密钥。
export function setSessionProfile(profile: {
  llm_base_url: string;
  llm_model: string;
  llm_api_key: string;
  llm_max_tokens: number;
  llm_temperature: number;
  llm_timeout: number;
}): Promise<Settings> {
  return request<Settings>(
    "/api/settings/session-profile",
    {
      method: "POST",
      body: JSON.stringify(profile),
    },
    { hideErrorDetail: true },
  );
}

// ---- 导出：三种格式都用 fetch + blob + <a download> 触发下载 ----

// 解析 Content-Disposition，优先取 filename*（RFC 5987，UTF-8 百分号编码），
// 兼容老客户端 fallback 的 filename=。中文文件名以前者为准。
export function parseContentDisposition(
  header: string | null
): string | null {
  if (!header) return null;
  // filename*=UTF-8''<percent-encoded>
  const star = header.match(/filename\*\s*=\s*UTF-8''([^;]+)/i);
  if (star && star[1]) {
    try {
      return decodeURIComponent(star[1].trim().replace(/^["']|["']$/g, ""));
    } catch {
      /* 解码失败回退到 filename= */
    }
  }
  const plain = header.match(/filename\s*=\s*("?)([^";]+)\1/i);
  if (plain && plain[2]) {
    return plain[2].trim().replace(/^["']|["']$/g, "");
  }
  return null;
}

export type ExportFormat = "md" | "html" | "pdf";

export async function exportDoc(pid: number, fmt: ExportFormat): Promise<void> {
  const res = await fetch(`/api/projects/${pid}/export/${fmt}`);
  if (!res.ok) {
    let msg = `导出失败 (${res.status})`;
    try {
      const err = await res.json();
      if (err && typeof err.detail === "string") msg = err.detail;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  const blob = await res.blob();
  const filename =
    parseContentDisposition(res.headers.get("Content-Disposition")) ??
    `knowledge.${fmt}`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // 释放要延后：click() 只是发起下载，浏览器可能还没开始读 blob。
  // 同一 tick 内 revokeObjectURL 会让 Safari / Firefox 静默取消下载
  //（Chrome 宽容，所以本地难复现），1 秒足够下载启动又不至于泄漏。
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const openLocalProject=(grant:string)=>request<Project>('/api/desktop/open',{method:'POST',body:JSON.stringify({grant})});
export const importLocalMedia=(pid:number,grant:string)=>request<Video>(`/api/desktop/projects/${pid}/media`,{method:'POST',body:JSON.stringify({grant})});
export const saveLocalProject=(pid:number)=>request<{status:string;message:string}>(`/api/desktop/projects/${pid}/save`,{method:'POST'});
export const projectTasks=(pid:number, options: {signal?: AbortSignal} = {})=>requestWithDeadline<ProjectTaskRecord[]>(`/api/projects/${pid}/tasks`, options.signal);
export function taskActivity(
  pid: number,
  task: TaskActivityIdentity,
  options: { signal?: AbortSignal } = {},
): Promise<TaskActivityResponse> {
  const taskId = typeof task.task_id === "string" ? task.task_id.trim() : "";
  let query: string;
  if (taskId) {
    query = `task_id=${encodeURIComponent(taskId)}`;
  } else if (
    Number.isSafeInteger(task.video_id)
    && Number(task.video_id) > 0
    && typeof task.recovery_id === "string"
    && /^[a-f\d]{24}$/i.test(task.recovery_id)
  ) {
    query = `video_id=${task.video_id}&recovery_id=${encodeURIComponent(task.recovery_id)}`;
  } else {
    return Promise.reject(new Error("工作流任务身份无效"));
  }
  return requestWithDeadline<TaskActivityResponse>(
    `/api/projects/${pid}/activity?${query}`,
    options.signal,
    "工作流读取超时，请稍后重试",
  );
}
export const taskDraft=(pid:number,id:string)=>request<KnowledgeDoc>(`/api/projects/${pid}/tasks/${encodeURIComponent(id)}/draft`);
export const resumeTask=(pid:number,videoId:number,recoveryId:string)=>request<{task_id:string}>(`/api/projects/${pid}/resume`,{method:'POST',body:JSON.stringify({video_id:videoId,recovery_id:recoveryId})});
export const rebuildTask=(pid:number,videoId:number,recoveryId:string)=>request<{task_id:string}>(`/api/projects/${pid}/rebuild`,{method:'POST',body:JSON.stringify({video_id:videoId,recovery_id:recoveryId})});

export const exportLocalDoc=(pid:number,fmt:ExportFormat)=>request<{path:string}>(`/api/desktop/projects/${pid}/export/${fmt}`,{method:"POST"});

export const testModelConnection = (body: Record<string,unknown>) => request<{ok:boolean;elapsed_ms:number}>('/api/settings/test', {method:'POST',body:JSON.stringify(body)});
