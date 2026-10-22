// 与后端响应对齐的类型定义（后端已完成，前端只消费）。
// 注意：GET /api/projects 列表项不含 videos，视频封面需走 listVideos。

export interface Project {
  id: number;
  title: string;
  subject?: string;
  created_at?: string;
  storage_root?: string;
}

export interface Video {
  id: number;
  bvid?: string | null;
  cid?: string | null;
  title?: string | null;
  duration?: number | null;
  cover?: string | null;
  // 后端落库的是英文阶段枚举（见 pipeline 的 STAGES：download/asr/draft/
  // frames/verify/illustrate/review/revise/render），外加 download 之前的 downloading 与终态 done/failed。
  // 这里把这些枚举都列出来；再补一个 `| string` 兜底——因为后端新增/改名阶段时，
  // ChatPanel 的 VIDEO_STATUS_LABELS 映射需要覆盖，漏覆盖时回退成原始枚举字符串
  // （见 ChatPanel「未知状态回退为原始枚举值」测试），而非让 tsc 把它当非法值卡死。
  status:
    | "pending"
    | "downloading"
    | "download"
    | "asr"
    | "draft"
    | "frames"
    | "verify"
    | "illustrate"
    | "review"
    | "revise"
    | "render"
    | "done"
    | "failed"
    | string;
  error?: string | null;
  source?: string;
  // 分P 元信息：仅 B站链接导入分支（POST /api/projects/{pid}/videos）返回，
  // 本地文件路径上传那条分支不返回，因此全为可选。
  // - page：实际导入第几 P
  // - total_pages：视频共几 P
  // - page_warning：分P 参数越界/语法无效被回退到 P1 时给出（意味着「你要的没拿到」）
  page?: number | null;
  total_pages?: number | null;
  page_warning?: string | null;
}

export type Playback = {
  title: string;
  source_url?: string;
  page?: number;
  storage?: "cache";
  cache_status?: "ready" | "missing" | "audio_only" | "video_only" | "unmerged" | "invalid";
  cache_message?: string;
} & ({kind: "embed" | "video" | "audio"; url: string} | {kind: "unavailable"; url: null});

export interface ActivitySnapshot {
 steps: AgentStep[];
 segments: {id:string;text:string;phase:string;order:number}[];
 started_at:number;
 elapsed_ms:number;
}
export interface Message {
  activity?: ActivitySnapshot;
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at?: string | null;
}

/**
 * 用户此刻在右侧知识面板的浏览位置。
 *
 * **这是「agent 反复反问已知信息」的根因修复**。用户报的真实症状是：
 * 他第 4 轮说「我说的 html 要改为 Apple 风格」，助手却反问
 * 「是苹果公司、苹果产品、苹果网站应用，还是一个网页？」——
 * 而他第 1 轮明明说过是做 iPhone/Mac 的产品。
 *
 * 模型在拿不到信息时反问是**合理的**，它不是笨，是架构没把信息送到。
 * 后端完全不知道用户正对着哪一块，只能靠对话猜作用域。
 *
 * 四个字段都允许为空：用户可能刚打开项目、还没点开任何知识点。
 * 全空时后端**整段都不写进提示词**（而不是写「未指定」——
 * 那等于明确告诉模型「这里缺信息」，反而诱发它反问）。
 */
export interface FocusRef {
  doc_title: string;
  point_title: string;
  module_title: string;
  view: string;
}

export interface Example {
  type?: "code" | "table" | "decision_tree" | "diagram" | "enum" | string;
  content: string;
  headers?: string[];
  rows?: string[][];
  nodes?: string[];
  layout?: "parallel" | "sequence";
  basis?: "source" | "illustrative";
  label?: string;
  source_quote?: string;
}

export interface KnowledgeFigure {
  id: string;
  kind: "video_frame";
  src: string;
  alt: string;
  caption: string;
  second: number;
  source_ref: string;
}

export interface Point {
  id: string;
  title: string;
  anchor?: string;
  body?: string;
  examples?: Example[];
  figures?: KnowledgeFigure[];
  figure_decision?: { needs_figure: boolean; needs_example: boolean; reason: string };
  warnings?: string[];
  mastery?: "new" | "learning" | "mastered" | string;
}

export interface Module {
  id: string;
  title: string;
  points: Point[];
}

export interface Cheatsheet {
  title?: string;
  headers?: string[];
  rows?: string[][];
}

/**
 * 后端设置（LLM 一组）。
 *
 * **没有 `llm_api_key` 字段，这是刻意的**：明文密钥永不过前端。
 * 只有 `llm_api_key_masked`（给用户认「是不是我那把」）
 * 与 `llm_api_key_set`（是否已配置；null 表示尚未读取钥匙串确认）。
 * 少一个字段名，就少一处将来被人顺手 `console.log` 打出密钥的机会。
 */
export interface Settings {
  llm_base_url: string;
  llm_model: string;
  llm_api_key_masked: string;
  llm_api_key_set: boolean | null;
  llm_api_key_status?: "configured" | "missing" | "unchecked";
  llm_api_key_scope?: "session" | "saved" | "unchecked";
  llm_profile_scope?: "session" | "saved";
  llm_max_tokens: number;
  llm_temperature: number;
  llm_timeout: number;
}

export interface IndependentReview {
  status: "stale" | "reviewed" | "revised" | string;
  scope?: string;
  limitations?: string[];
  findings?: { id?: string; point_id?: string; issue: string; suggestion?: string }[];
  decisions?: { finding_id?: string; decision: "accept" | "reject" | "defer" | string; reason: string }[];
  web_sources?: { url: string; title?: string }[];
}

export interface KnowledgeDoc {
  meta?: { title?: string; positioning?: string; visual_review?: {status: string; reason?: string}; verification?: {status: string; note?: string}; independent_review?: IndependentReview };
  modules?: Module[];
  appendix?: { cheatsheets?: Cheatsheet[] };
}

// agent 活动事件里的一步（后端 ChatAgent._emit_step 发出）。
//
// 为什么是结构化对象而不是一句文案：旧实现全程只有「思考中」三个字，
// 12 秒里 UI 只变化 2 次（实测抓到的轨迹是 20% → 50% → 消失），
// 用户完全看不出 agent 在干什么。信息必须在后端就拆开。
export interface AgentStep {
  order?: number;
  // 严格递增的事件序号，前端靠它判断「这是第几步」
  seq: number;
  // 步骤标识（load_history / build_prompt / llm_call / tool_call / …）
  step: string;
  // 给人看的中文标签，如「检索知识点」
  label: string;
  // 状态可为 pending/running/done/failed/cancelled/interrupted，保留 string 兼容未来阶段。
  status: "pending" | "running" | "done" | string;
  // 结果摘要，如「3 条」。失败时是「失败：…」
  detail?: string;
  // 该步真实耗时（毫秒）。长步骤显示秒数是消除「卡死感」最有效的一招
  elapsed_ms?: number | null;
  actor?: "orchestrator" | "author" | "reviewer";
  agent_id?: string;
  parent_task_id?: string;
  child_task_id?: string;
  stage?: string;
  round?: number;
  cycle?: number;
  attempts?: number;
  retry_count?: number;
  tool_name?: string;
  arguments?: unknown;
  result_summary?: string | null;
  error?: string | null;
  model?: string;
  finish_reason?: string | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  reasoning_tokens?: number | null;
  total_tokens?: number | null;
  max_tokens?: number | null;
  http_status?: number | null;
  started_at?: number | string | null;
  finished_at?: number | string | null;
  execution?: "isolated_context" | "child_task" | "workflow";
  persistence_warning?: string | null;
  truncated?: boolean;
}

// SSE 事件负载（来自后端队列任务字典，经 event_stream 透传）
export interface TaskEvent {
  activity?: ActivitySnapshot;
  kind?: string;
  draft_available?: boolean;
  created_at?: string | null;
  source_recovery_id?: string | null;
  source_task_id?: string | null;
  // Persistent pipeline recovery metadata. recovery_id identifies a disk artifact,
  // not this queue task, and is supplied only when the server allows resuming.
  video_id?: number;
  recovery_id?: string;
  resumable?: boolean;
  rebuildable?: boolean;
  recovery_reason?: string;
  task_id: string;
  // cancelled 是第三种终态，与 done / failed 并列。
  // 为什么不复用 failed：用户主动停止不是错误，复用 failed 会让前端
  // 弹一条红色报错条，把用户的正确操作报成了故障。
  status: "pending" | "running" | "done" | "failed" | "cancelled" | string;
  progress: number; // 0..1
  message?: string;
  stage?: string; // download | frames | asr | draft | verify | illustrate | review | revise | render | chat
  result?: unknown;
  error?: string;
  // 后端 cancel() 置位。区别于 `status === "cancelled"`：
  // 任务若在取消请求到达前就已完成，status 仍是 done，此时这个标志也会是 true
  // （后端把「停止一个已完成的任务」当作 no-op）。
  cancelled?: boolean;
  // 事件类型分流：step=活动流水，token=流式正文片段，waiting=任务尚未创建
  //
  // 前端按这个字段分流，而不是「看有没有 step 字段」——后者在 waiting
  // 事件上会误判（它没有 step 但也不是别的类型）。
  event?: "step" | "token" | "waiting" | string;
  step?: AgentStep;
  token?: string;
}

// 项目任务列表也可能包含旧的持久化恢复记录；这些没有队列 task_id，
// 但仍可通过 video_id + recovery_id 定位其活动快照。
export type ProjectTaskRecord = Omit<TaskEvent, "task_id"> & { task_id: string | null };

export interface TaskActivityIdentity {
  task_id?: string | null;
  video_id?: number;
  recovery_id?: string;
}

export interface TaskActivityResponse {
  task_id: string | null;
  status: string;
  stage?: string | null;
  message?: string | null;
  error?: string | null;
  activity: ActivitySnapshot | null;
  activity_available: boolean;
  activity_source: "live" | "persisted" | "unavailable";
  model_budget?: Record<string, unknown> | null;
  activity_warning?: string | null;
  activity_persistence_error?: string | null;
  activity_truncated?: boolean;
}

export interface ProcessingOptions {
  document_mode: 'recitation' | 'textbook' | null;
  integration: 'merge' | 'replace';
  example_strategy: 'as_needed' | 'course' | 'analogy' | null;
  hard_subtitles: boolean;
  subtitle_crop_bottom: number;
}
