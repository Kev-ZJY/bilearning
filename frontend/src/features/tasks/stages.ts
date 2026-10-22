import type { TaskEvent } from "../../api/types";

// 对应后端实际执行阶段：完成转写与音画分析后起草，再补图、审查、修订。
const STAGE_LABELS: Record<string, string> = {
  download: "下载音频",
  frames: "音画联合分析",
  asr: "语音转写",
  draft: "起草知识点",
  verify: "核对来源与覆盖",
  illustrate: "补充图例",
  review: "独立审查",
  revise: "修订知识文档",
  render: "渲染",
  // 对话任务（ChatAgent）用的是同一个 SSE 通道，但阶段名是 "chat"。
  // 不映射的话用户会看到裸枚举「chat · 思考中」。
  chat: "对话",
};

export function mapStageToLabel(stage?: string): string {
  if (!stage) return "";
  return STAGE_LABELS[stage] ?? stage;
}

// 把 SSE 事件里要展示的状态文案拼好：阶段 + 后端 message。
export function describeTask(task: TaskEvent | null): string {
  if (!task) return "";
  const label = mapStageToLabel(task.stage);
  const msg = task.message ?? "";
  if (label && msg) return `${label} · ${msg}`;
  return label || msg;
}
