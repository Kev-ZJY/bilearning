import { useCallback, useEffect, useRef, useState } from "react";
import type { AgentStep, TaskEvent } from "../../api/types";

// SSE 进度订阅（前端最核心交互）。
// - 连 /api/tasks/{task_id}/events
// - status=done 必须关闭 EventSource（否则连接泄漏），并回调 onComplete 刷新知识面板
// - status=failed 展示 error，并把视频标记为失败（由父组件据 video_id 处理）
// - status=cancelled 关流并清流式正文，但**不弹错误**（那是用户自己按的停止）
// - 降级：长时间「没收到事件且进度确实卡住」才标记 stale，提示用户去刷新，
//   绝不卡死界面。EventSource 自带自动重连，无法连接时也不抛未捕获异常。
//
// 三类事件分别累积，不互相覆盖
// --------------------------------
// 后端会推三种 event，前端各有各的渲染路径，所以必须分开存：
// - `step`  ：活动流水（agent 此刻在做哪一步）。**累积成数组**，
//             每次事件 push 一条。按 seq 排序渲染成列表。
// - `token` ：流式正文片段。**累积成字符串**，前端边收边显示「字在长出来」。
// - 其他   ：粗粒度任务状态（进度条），保留最后一个即可（progress 是快照）。
//
// 踩过的坑：早期只保留 `task` 这一个快照对象，于是 step 事件每来一个
// 就把上一个覆盖掉，活动流永远只显示一行——而后端明明一直在推。
// 「快照」和「流水」是两种数据模型，用一个 state 变量装不下。
//
// 多久「既没收到事件、进度又卡住」才算断流。
//
// 为什么改成看进度而不是看事件：后端 ASR 阶段现在会持续回传「转写中 X%」，
// 长视频（如 41 小时的课程）这个区间会持续数小时。旧逻辑「90 秒没收到任何事件
// 就告警」会在这类长阶段全程误报，用户一直盯着一条「进度更新暂时中断」的假警告。
//
// 新逻辑：只在「长时间没收到事件 **且** 进度与上一个事件相同（即真卡住）」时才
// 告警。因为现在长阶段本来就有持续事件，真正能触发这里的只能是「流真的断了」，
// 所以阈值放宽到 10 分钟，退化成一个兜底的断流探测，而不是正常长阶段的噪音。
const STALE_TIMEOUT_MS = 10 * 60_000; // 10 分钟
const WATCHDOG_TICK_MS = 5_000;

function stepIdentity(step: AgentStep): string {
  return [
    step.actor ?? "",
    step.agent_id ?? "",
    step.parent_task_id ?? "",
    step.child_task_id ?? "",
    step.seq,
  ].join("\u0000");
}

function normalizeStep(step: AgentStep): AgentStep {
  const maxTokens = step.max_tokens;
  if (
    typeof maxTokens === "number"
    && Number.isSafeInteger(maxTokens)
    && maxTokens >= 0
    && maxTokens <= 1_000_000_000
  ) {
    return step;
  }
  const { max_tokens: _invalid, ...safe } = step;
  return safe;
}

export function useTaskEvents(onComplete?: (task: TaskEvent) => void) {
  const [task, setTask] = useState<TaskEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);
  // agent 活动流水：累积而非覆盖（见文件头说明）
  const [steps, setSteps] = useState<AgentStep[]>([]);
  // 流式正文：边收边显示
  const [streamText, setStreamText] = useState("");

  const esRef = useRef<EventSource | null>(null);
  const lastUpdateRef = useRef<number>(0);
  // 最近一次带 progress 的事件进度，以及它「上一个」事件的进度。
  // 只有两者相等（进度没推进）且长时间没事件，才说明流真断了，而不是长阶段正常静默。
  const lastProgressRef = useRef<number | null>(null);
  const prevProgressRef = useRef<number | null>(null);
  const watchdogRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const clearWatchdog = useCallback(() => {
    if (watchdogRef.current != null) {
      clearInterval(watchdogRef.current);
      watchdogRef.current = null;
    }
  }, []);

  /**
   * 立刻清掉流式正文。由上层在**正式气泡真的上屏之后**调用。
   *
   * 为什么不能由本 hook 在 done 时自动清：done 帧到达时回复虽已落库，
   * 但气泡要等 onComplete → refreshProject 走完一次网络往返才渲染。
   * 提前清会在两者之间留一个空窗——正文消失、气泡未现，看着像答案被吞。
   * 短答案尤其明显：实测只有 2 条 token、间隔 255ms，done 紧跟第二条，
   * 用户看到的是 20 个字闪一下就没了，压根没读到完整答案。
   *
   * 为什么不在这里用 setTimeout 兜一下：定时器只是把「清空」推后一个
   * 微任务，仍然早于那次网络往返，空窗原样存在。**只有上层知道
   * 气泡什么时候真的画出来了**，所以决定权必须交上去。
   */
  const clearStream = useCallback(() => setStreamText(""), []);

  const close = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    clearWatchdog();
    setStale(false);
  }, [clearWatchdog]);

  // 切换项目时同时解除旧任务订阅，后台任务继续运行。
  // 否则旧项目的下一帧 SSE 会把新项目重新标成忙碌。
  const reset = useCallback(() => {
    close();
    setTask(null);
    setError(null);
    setStale(false);
    setSteps([]);
    setStreamText("");
  }, [close]);

  const connect = useCallback(
    (taskId: string) => {
      // 先清理可能存在的旧连接
      close();
      setTask(null);
      setError(null);
      setStale(false);
      // 新任务开始：清空上一轮的活动流水和流式正文。
      // 不清的话上一轮的步骤会留在界面上，用户以为这轮已经做完了。
      setSteps([]);
      setStreamText("");
      lastUpdateRef.current = Date.now();
      lastProgressRef.current = null;
      prevProgressRef.current = null;

      const es = new EventSource(`/api/tasks/${taskId}/events`);
      esRef.current = es;
      lastUpdateRef.current = Date.now();

      // 看门狗：没收到事件且进度卡住（与上一事件相同）才进入降级提示。
      // 阈值放宽到 10 分钟：现在长阶段本来就有持续进度事件，正常处理绝不会触发，
      // 触发的只能是「流真的断了」这种兜底情况。
      watchdogRef.current = setInterval(() => {
        const silent = Date.now() - lastUpdateRef.current > STALE_TIMEOUT_MS;
        const last = lastProgressRef.current;
        const prev = prevProgressRef.current;
        // 从未收到带 progress 的事件（last 为 null）也算「没进度」，
        // 或进度与上一事件相同（说明这段时间没推进）——两者任一 + 静默即告警。
        const progressStuck = last === null || last === prev;
        if (silent && progressStuck) {
          setStale(true);
        }
      }, WATCHDOG_TICK_MS);

      es.onmessage = (ev) => {
        if (esRef.current !== es) return;
        lastUpdateRef.current = Date.now();
        try {
          const data = JSON.parse(ev.data) as TaskEvent;
          // 只在带 progress 的事件上更新进度记录：没有 progress 的连接
          // 不会因此误判为「进度卡住」。
          if (typeof data.progress === "number") {
            prevProgressRef.current = lastProgressRef.current;
            lastProgressRef.current = data.progress;
          }
          setStale(false);

          // 按 event 字段分流。三类事件各有渲染路径，塞在一起会互相覆盖。
          if (data.activity) {
            const steps = data.activity.steps.map(normalizeStep);
            data.activity = { ...data.activity, steps };
          } else if (data.event === "step" && data.step) {
            data.step = normalizeStep(data.step);
          }

          if (data.activity) {
            const steps = data.activity.steps;
            setSteps(steps);
            // A pending segment may still become tool commentary after the model finishes.
            setStreamText(data.activity.segments.filter(s=>s.phase==='final').map(s=>s.text).join('\n'));
          } else if (data.event === "step" && data.step) {
            // 活动流水：累积。同一 step 的重复上报（running → done）
            // 要**替换**而不是追加，否则同一步会出现两行。
            setSteps((cur) => {
              const s = data.step as AgentStep;
              // 同 seq 是同一步的状态更新；同 (step,label) 但 seq 不同
              // 是新一轮的同名步骤（多轮工具循环），必须分开显示。
              const identity = stepIdentity(s);
              const i = cur.findIndex((x) => stepIdentity(x) === identity);
              if (i >= 0) {
                const next = [...cur];
                next[i] = s;
                return next;
              }
              return [...cur, s];
            });
          } else if (data.event === "token" && typeof data.token === "string") {
            // 流式正文：追加
            setStreamText((cur) => cur + data.token);
          }

          setTask(data);
          if (data.status === "done") {
            // **不在这里清流式正文。**
            //
            // 落库后的正式回复由 onComplete → refreshProject 拉回来渲染成
            // 正式气泡，那要一次网络往返。此刻清掉流式正文会留下一个空窗：
            // 正文消失、气泡未现，界面看着像答案被吞了。
            //
            // 实测（diag_stream_layer.mjs hook EventSource 抓的原始帧）：
            // 短答案只有 2 条 token、间隔 255ms，done 帧紧跟第二条。清空
            // 与追加被 React 批进同一次渲染，DOM 上只剩第一条的 20 个字
            // 闪一下就没了 —— 用户压根没读到完整答案。
            //
            // 真正的清空时机是「正式气泡已上屏」，只有上层知道，
            // 所以由 App 在 messages 到位后调 clearStream()。
            // steps 保留 —— 它们是「agent 干了哪几步」的唯一留痕。
            close();
            onCompleteRef.current?.(data);
          } else if (data.status === "cancelled") {
            // 用户自己按的停止。**刻意不 setError**：那是红色报错条，
            // 会把「我刚点的停止」报成故障。
            //
            // 流式正文要清：被取消的回答不会落库（后端 run() 对
            // TaskCancelled 单独处理，不写 assistant 消息），所以不会有
            // 正式气泡来替换它 —— 留着就是一段永远停在半截的正文。
            // 与 failed 同样处理，但**不弹错**。
            setStreamText("");
            close();
          } else if (data.status === "failed") {
            // 失败时**立即**清：不会有正式气泡，正文留着只会误导
            // （用户以为答案已经写好了，只是没显示出来）。
            setStreamText("");
            close();
            setError(data.error ?? "处理失败");
          }
        } catch {
          // 解析失败的单条事件直接忽略，不中断流
        }
      };

      es.onerror = () => {
        // 连接异常（后端不可达 / 流被异常中断）。若我们已主动关闭则不处理。
        //
        // 这里**必须**给出可见反馈，不能只置 stale：stale 的阈值是 10 分钟，
        // 也就是说连接一断，用户要干等 10 分钟才看到「可能出问题了」，
        // 而这 10 分钟里界面完全静止——实测用户就在这个窗口里以为
        // 「agent 挂了」。断开是**立刻**告诉用户的事，不是 10 分钟后。
        if (esRef.current && esRef.current.readyState !== EventSource.CLOSED) {
          setStale(true);
        }
      };
    },
    [close]
  );

  // 卸载时清理，避免 EventSource / 定时器泄漏
  useEffect(() => () => close(), [close]);

  return { task, error, stale, steps, streamText, connect, close, reset, clearStream };
}
