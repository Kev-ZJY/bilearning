import type { Message, ProjectTaskRecord } from "../../api/types";

export interface ConversationMessageEntry {
  kind: "message";
  key: string;
  created_at: string | null;
  message: Message;
}

export interface ConversationTaskEntry {
  kind: "task_group";
  key: string;
  created_at: string | null;
  source_recovery_id: string | null;
  attempts: ProjectTaskRecord[];
  latest: ProjectTaskRecord;
}

export type ConversationTimelineEntry = ConversationMessageEntry | ConversationTaskEntry;

function timestamp(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function videoKey(task: ProjectTaskRecord): string | null {
  return typeof task.video_id === "number" && Number.isInteger(task.video_id) && task.video_id > 0
    ? String(task.video_id)
    : null;
}

function recoveryKey(value: string | null | undefined): string | null {
  const normalized = value?.trim().toLowerCase();
  return normalized ? normalized : null;
}

function stableTaskKey(task: ProjectTaskRecord, index: number): string {
  if (typeof task.task_id === "string" && task.task_id) return `task:${task.task_id}`;
  const video = videoKey(task);
  if (typeof task.source_task_id === "string" && task.source_task_id) {
    return `source-task:${video ?? "unknown-video"}:${task.source_task_id}`;
  }
  const ownRecoveryId = recoveryKey(task.recovery_id);
  if (video && ownRecoveryId) return `recovery:${video}:${ownRecoveryId}`;
  return `task-index:${index}`;
}

function mergeDuplicateTaskRecords(tasks: ProjectTaskRecord[]): ProjectTaskRecord[] {
  const positions = new Map<string, number>();
  const merged: ProjectTaskRecord[] = [];
  tasks.forEach((task, index) => {
    const key = stableTaskKey(task, index);
    const previousIndex = positions.get(key);
    if (previousIndex === undefined) {
      positions.set(key, merged.length);
      merged.push(task);
      return;
    }
    const previous = merged[previousIndex];
    merged[previousIndex] = {
      ...previous,
      ...task,
      created_at: task.created_at ?? previous.created_at,
      source_recovery_id: task.source_recovery_id ?? previous.source_recovery_id,
      source_task_id: task.source_task_id ?? previous.source_task_id,
      activity: task.activity ?? previous.activity,
    };
  });
  return merged;
}

/**
 * Build the conversation's chronological entries from actual saved messages and
 * task records. It deliberately does not invent messages or timestamps.
 */
export function buildConversationTimeline(
  messages: Message[],
  taskRecords: ProjectTaskRecord[],
  representedTaskIds: string[] = [],
): ConversationTimelineEntry[] {
  const represented = new Set(representedTaskIds.filter(Boolean));
  for (const message of messages) {
    for (const step of message.activity?.steps ?? []) {
      if (typeof step.child_task_id === "string" && step.child_task_id) represented.add(step.child_task_id);
    }
  }

  const allTasks = mergeDuplicateTaskRecords(taskRecords);
  const visibleTasks = allTasks.filter(task => {
    return ![task.task_id, task.source_task_id].some(id => typeof id === "string" && represented.has(id));
  });
  const allTaskIndex = new Map(allTasks.map((task, index) => [task, index]));

  const recoveryRecords = new Map<string, number[]>();
  allTasks.forEach((task, index) => {
    const video = videoKey(task);
    const ownRecoveryId = recoveryKey(task.recovery_id);
    if (!video || !ownRecoveryId) return;
    const key = `${video}:${ownRecoveryId}`;
    const matches = recoveryRecords.get(key) ?? [];
    matches.push(index);
    recoveryRecords.set(key, matches);
  });

  const lineageFor = (task: ProjectTaskRecord, index: number): { key: string; sourceRecoveryId: string | null } => {
    const video = videoKey(task);
    if (!video) return { key: `single:${stableTaskKey(task, index)}`, sourceRecoveryId: null };
    const isolatedRecord = () => ({
      key: `isolated:${video}:${stableTaskKey(task, index)}`,
      sourceRecoveryId: null,
    });

    let current = task;
    let currentIndex = index;
    const chain: Array<{ nodeKey: string; recoveryId: string | null }> = [];
    const seen = new Map<string, number>();
    while (true) {
      const currentRecoveryId = recoveryKey(current.recovery_id);
      const nodeKey = currentRecoveryId
        ? `${video}:${currentRecoveryId}`
        : `task:${stableTaskKey(current, currentIndex)}`;
      const cycleStart = seen.get(nodeKey);
      if (cycleStart !== undefined) {
        return isolatedRecord();
      }
      seen.set(nodeKey, chain.length);
      chain.push({ nodeKey, recoveryId: currentRecoveryId });

      const parentRecoveryId = recoveryKey(current.source_recovery_id);
      if (!parentRecoveryId) {
        return currentRecoveryId
          ? { key: `lineage:${video}:${currentRecoveryId}`, sourceRecoveryId: currentRecoveryId }
          : { key: `lineage:${video}:${stableTaskKey(current, currentIndex)}`, sourceRecoveryId: null };
      }

      const parents = recoveryRecords.get(`${video}:${parentRecoveryId}`) ?? [];
      // An absent or ambiguous parent cannot establish a root. Keep its exact
      // record isolated; another record pointing at the same missing parent
      // does not prove that the two records belong to the same lineage.
      if (parents.length !== 1) {
        return isolatedRecord();
      }
      currentIndex = parents[0];
      current = allTasks[currentIndex];
    }
  };

  const groups = new Map<string, { sourceRecoveryId: string | null; tasks: ProjectTaskRecord[]; firstIndex: number }>();
  visibleTasks.forEach((task, index) => {
    const lineage = lineageFor(task, allTaskIndex.get(task) ?? index);
    const existing = groups.get(lineage.key);
    if (existing) existing.tasks.push(task);
    else groups.set(lineage.key, { sourceRecoveryId: lineage.sourceRecoveryId, tasks: [task], firstIndex: index });
  });

  const entries: Array<{ entry: ConversationTimelineEntry; index: number; sortPriority: number }> = [];
  messages.forEach((message, index) => {
    entries.push({
      entry: {
        kind: "message",
        key: `message:${message.id}`,
        created_at: message.created_at ?? null,
        message,
      },
      index,
      sortPriority: message.role === "user" ? 0 : 2,
    });
  });

  for (const [key, group] of groups) {
    const indexedAttempts = group.tasks.map((task, index) => ({ task, index, time: timestamp(task.created_at) }));
    indexedAttempts.sort((a, b) => {
      if (a.time !== null && b.time !== null && a.time !== b.time) return a.time - b.time;
      if (a.time !== null && b.time === null) return -1;
      if (a.time === null && b.time !== null) return 1;
      return a.index - b.index;
    });
    const attempts = indexedAttempts.map(item => item.task);
    const parentRecoveryIds = new Set(indexedAttempts
      .map(item => recoveryKey(item.task.source_recovery_id))
      .filter((id): id is string => id !== null));
    const leaves = indexedAttempts.filter(item => {
      const ownRecoveryId = recoveryKey(item.task.recovery_id);
      return !ownRecoveryId || !parentRecoveryIds.has(ownRecoveryId);
    });
    const latestDatedLeaf = [...leaves].reverse().find(item => item.time !== null)?.task;
    const latest = latestDatedLeaf ?? leaves[leaves.length - 1]?.task ?? attempts[attempts.length - 1];
    const datedAttempts = indexedAttempts.filter(item => item.time !== null);
    const createdAt = datedAttempts[0]?.task.created_at ?? null;
    entries.push({
      entry: {
        kind: "task_group",
        key,
        created_at: createdAt,
        source_recovery_id: group.sourceRecoveryId,
        attempts,
        latest,
      },
      index: messages.length + group.firstIndex,
      sortPriority: 1,
    });
  }

  // Unknown timestamps stay unknown and follow dated entries in stable source
  // order. Equal known timestamps place user input before work and its reply,
  // without claiming a more precise timestamp than the API supplied.
  entries.sort((a, b) => {
    const aTime = timestamp(a.entry.created_at);
    const bTime = timestamp(b.entry.created_at);
    if (aTime !== null && bTime !== null && aTime !== bTime) return aTime - bTime;
    if (aTime !== null && bTime === null) return -1;
    if (aTime === null && bTime !== null) return 1;
    if (aTime === null && bTime === null) return a.index - b.index;
    return a.sortPriority - b.sortPriority || a.index - b.index;
  });
  return entries.map(item => item.entry);
}
