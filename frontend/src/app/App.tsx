import { useCallback, useEffect, useRef, useState } from "react";
import {isDesktop,selectLocal,revealProject} from "../api/desktop";
import {ProjectTasks} from "../features/tasks/ProjectTasks";
import { Sidebar } from "../features/projects/Sidebar";
import {videoRequests} from "../features/chat/composer";
import { ChatPanel, formatVideoPageInfo } from "../features/chat/ChatPanel";
import { KnowledgePanel } from "../features/knowledge/KnowledgePanel";
import { ProcessingDialog } from "../features/video/ProcessingDialog";
import { SettingsDialog } from "../features/settings/SettingsDialog";
import { VideoPlayer } from "../features/video/VideoPlayer";
import * as api from "../api/client";
import { useTaskEvents } from "../features/tasks/useTaskEvents";
import type {
  ProcessingOptions,
  FocusRef,
  KnowledgeDoc,
  Message,
  Project,
  Video,
} from "../api/types";

const VIDEO_PIPELINE_STAGES = new Set([
  "processing", "downloading", "download", "asr", "draft", "frames",
  "verify", "illustrate", "review", "revise", "render",
]);

export default function App() {
  const [workspaceView, setWorkspaceView] = useState<"chat" | "video">("chat");
  const [selectedVideoId, setSelectedVideoId] = useState<number | null>(null);
  const [previewDraft,setPreviewDraft]=useState<KnowledgeDoc|null>(null);
  const [storageStatus,setStorageStatus]=useState<{status:string;message:string}|null>(null);
  const [storageRoot,setStorageRoot]=useState('');
  const [readerOpen,setReaderOpen]=useState(true);
  const [existingDocument,setExistingDocument]=useState(false);
  const [showOriginal,setShowOriginal]=useState(false);
  const [menuOpen,setMenuOpen]=useState(false);
  const [panelWidth,setPanelWidth]=useState(50);
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;
  const [messages, setMessages] = useState<Message[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [processingVideo, setProcessingVideo] = useState<{projectId:number;videoId:number}|null>(null);
  const [doc, setDoc] = useState<KnowledgeDoc | null>(null);
  // 一次性提示独立于 messages：refreshProject 会全量替换 messages，
  // 错误若混在里面，下一次刷新就会把它抹掉。
  const [notice, setNotice] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  // 「已按下发送、还在等后端返回 task_id」。
  //
  // 为什么单独要这个 state：等 task_id 的那 2.2 秒里还没有 task_id，
  // 也就没有 task 对象，活动流无从判断「该不该显示」。用户看到的是
  // 完全静止的界面——这就是「问个问题又不响应」的真正成因。
  // 这里在**发请求之前**置 true，让占位立刻出现，不依赖任何网络往返。
  const [submitting, setSubmitting] = useState(false);
  const pendingSend = useRef<{pid:number;cancelled:boolean}|null>(null);
  // 最近一次提交拿到的 task_id。中止需要它 —— 停止按钮要打的是
  // 「当前这个任务」，而 useTaskEvents 只给了我们一个不含 id 的 task 快照
  // （TaskEvent 里有 task_id，但它只在有 SSE 连接时才有值；
  // 提交后、connect 前的那个窗口里前端手上是空的）。
  const [lastTaskId, setLastTaskId] = useState<string | null>(null);
  // 用户此刻在右侧面板停留的知识点。
  //
  // 为什么要单独用 ref 而不是 state：它要在**发送那一刻**被读出来，
  // 而不是驱动渲染。放 state 会让每次滚动右侧面板都重渲整个 App
  // （聊天区几百条消息全部重新 diff），而它唯一的用途是被 send 读一次。
  // 用 ref 读到的永远是最新值，且零渲染成本。
  //
  // 语义由 KnowledgePanel 上报，见它的 onFocusChange 注释。
  const focusRef = useRef<FocusRef | null>(null);
  // 消息输入框的内容。提到 App 层是为了让「撤回」能把原文填回去
  // （输入框原本是 ChatPanel 的内部 state，App 改不了）。
  const [draft, setDraft] = useState("");
  // 后端设置面板的开关。放在 App 而不是 Sidebar 内部：
  // 面板要用的 setNotice 也在这里，且「设置保存成功」应当出现在聊天区
  // —— 项目列表区是导航位置，塞业务提示会让人以为是某个项目出错了。
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [modelName,setModelName]=useState('');
  const [modelSettingsStatus,setModelSettingsStatus]=useState<"loading"|"ready"|"error">("loading");
  useEffect(()=>{
    let alive=true;
    setModelSettingsStatus("loading");
    api.getSettings().then(s=>{
      if(!alive)return;
      setModelName(s.llm_model);
      setModelSettingsStatus("ready");
    }).catch(()=>{if(alive)setModelSettingsStatus("error")});
    return ()=>{alive=false};
  },[settingsOpen]);

  // 拉取项目详情 + 视频列表（含封面）；视频列表用 listVideos 而非 getProject。
  const refreshProject = useCallback(async (pid: number) => {
    const [d, vs] = await Promise.all([
      api.getProject(pid),
      api.listVideos(pid),
    ]);
    if(activeIdRef.current!==pid)return;
    setStorageStatus(d.storage_status??null);setStorageRoot(d.storage_root??'');
    setExistingDocument(!!d.existing_document);
    setMessages(d.messages);
    api.listProjects().then(setProjects).catch(()=>{});
    setDoc(d.knowledge);
    setVideos(vs);
  }, []);

  // SSE 订阅：done 时刷新知识面板。
  // steps / streamText 是活动流与流式正文的数据源，必须从 hook 里取出来
  // 往下传——只传 task 的话，agent 的每一步都传不到 ChatPanel。
  const {
    task,
    error: taskError,
    stale,
    steps,
    streamText,
    connect,
    reset,
    close,
    clearStream,
  } = useTaskEvents(
    useCallback(() => {
      if (activeId != null) refreshProject(activeId);
    }, [activeId, refreshProject])
  );

  // A live pipeline task is authoritative. With no pipeline task, mirror
  // VideoPlayer's selected-video fallback so historical/imported video state
  // can still explain an empty knowledge panel. A running chat task must not
  // make the video workflow look active.
  const selectedVideo = videos.find((video) => video.id === selectedVideoId) ?? videos[0];
  const videoTaskStatus = selectedVideo?.status === "failed"
    ? "failed"
    : VIDEO_PIPELINE_STAGES.has(selectedVideo?.status ?? "")
      ? "running"
      : null;
  const knowledgeTaskStatus = task?.kind === "pipeline"
    ? task.status
    : videoTaskStatus;

  // 正式气泡上屏之后才清流式正文。
  //
  // 为什么不能由 useTaskEvents 在 done 时自己清：done 帧到达时回复虽已落库，
  // 但气泡要等上面那次 refreshProject 的网络往返才渲染。提前清会留一个
  // 空窗——正文消失、气泡未现，看着像答案被吞。实测短答案只有 2 条 token
  // （间隔 255ms），done 紧跟其后，同一次渲染里既追加又清空，
  // 用户看到的就是 20 个字闪一下消失。
  //
  // 判据用「messages 数量变多」而不是比对内容：乐观上屏的用户消息
  // 已经先放进去了，所以 assistant 气泡出现必然让长度增加。
  // 依赖内容比对会在 SSE 与 pollForAssistantReply 两条路径同时更新时误判。
  const msgCountRef = useRef(0);
  useEffect(() => {
    if (messages.length > msgCountRef.current) {
      msgCountRef.current = messages.length;
      clearStream();
    }
  }, [messages, clearStream]);

  useEffect(() => {
    api
      .listProjects()
      .then(setProjects)
      .catch((e) => setListError(`无法连接后端：${msg(e)}`));
  }, []);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      setVideos([]);
      setDoc(null);
      msgCountRef.current = 0;
      return;
    }
    // 切换项目时清掉上个项目的任务态，否则进度泡会串到新项目的聊天流里。
    reset();
    setWorkspaceView("chat"); setSelectedVideoId(null);
    setMessages([]); setVideos([]);
    setPreviewDraft(null);setStorageStatus(null);setStorageRoot('');
    setDoc(null);setExistingDocument(false);setShowOriginal(false);setReaderOpen(true);setMenuOpen(false);setDraft('');setLastTaskId(null);setSubmitting(false);focusRef.current=null;
    setNotice(null);
    // 基线重置到 0：新项目的消息数从头算，否则「变多」这个判据
    // 在新项目消息更少时会一直不成立，流式正文永远清不掉。
    msgCountRef.current = 0;
    refreshProject(activeId).catch((e) =>
      setNotice(`加载项目失败：${msg(e)}`)
    );
  }, [activeId, refreshProject, reset]);

  // 任务失败也要把视频状态刷成 failed（后端已写库）。
  useEffect(() => {
    if (task?.status === "failed" && activeId != null) {
      api.listVideos(activeId).then(setVideos).catch(() => {});
    }
  }, [task?.status, activeId]);

  const createProject = async (title: string, subject: string, directoryGrant?:string) => {
    setNotice(null);
    try {
      const p = await api.createProject({ title, subject, directory_grant:directoryGrant });
      setProjects(await api.listProjects());
      setActiveId(p.id);
      if(activeIdRef.current===p.id)await refreshProject(p.id);
    } catch (e) {
      // 注意：不能塞进 messages —— 此时多半还没有选中项目，
      // ChatPanel 会走空状态分支，messages 根本不在渲染树里。
      throw new Error(`创建失败：${msg(e)}`);
    }
  };

  // 点已选中的项目时主动刷新一次。直接传 setActiveId 的话，
  // React 对相同值会 bail out，useEffect 不重跑，点了等于没点。
  const selectProject = useCallback(
    (id: number) => {
      if (id === activeId) {
        refreshProject(id).catch((e) => setNotice(`刷新失败：${msg(e)}`));
        return;
      }
      setActiveId(id);
    },
    [activeId, refreshProject]
  );

  const process = async (intent: string, options: ProcessingOptions) => {
    if (!processingVideo) return;
    const {projectId,videoId}=processingVideo;
    const { task_id } = await api.process(projectId, { video_id: videoId, intent, options });
    if (activeIdRef.current === projectId) {
      setNotice(null);
      setLastTaskId(task_id);
      connect(task_id);
      try {
        const updated = await api.listVideos(projectId);
        if (activeIdRef.current === projectId) setVideos(updated);
      } catch { setNotice("整理已提交，视频列表刷新失败，可稍后刷新。"); }
    }
  };

  const send = async (text: string) => {
    if (activeId == null || !text.trim()) return;
    const pid = activeId;
    const mine = text;
    const focus = focusRef.current;
    const submission = {pid, cancelled:false};
    pendingSend.current = submission;
    setLastTaskId(null);
    close();
    reset();
    let requests: ReturnType<typeof videoRequests>;
    try { requests = videoRequests(text); }
    catch(e) {setNotice(msg(e)); return false;}

    // **先亮占位，再等网络。** 这是「点发送后界面有反应」的关键。
    //
    // 实测（Playwright 真实计时）：`await api.sendMessage()` 这一段要
    // 2245ms —— 服务端要落库、起线程、返回 task_id。等到 task_id 回来
    // 才 connect，首帧就晚了 2.2 秒。用户在这 2.2 秒里看到的是「我发了，
    // 什么都没发生」，于是又变成「问个问题就不响应」。
    //
    // 提交成功后 connect(task_id) 会清掉这个占位（connect 里
    // `setSteps([])` / `setStreamText("")`），所以两者不会叠加。
    setSubmitting(true);
    try {
      // 后端是异步的：立刻返回 task_id，agent 在后台跑工具循环。
      // 必须 connect(task_id)，否则用户在十几秒里只能看着自己那条消息
      // 干等，没有任何「在跑」的迹象——这正是当初被投诉的静默。
      const attachedVideoIds: number[] = [];
      const failures: string[] = [];
      const notes: string[] = [];
      for (const request of requests) {
        if(submission.cancelled)break;
        try {
          const video = await api.addVideo(pid, {...request, reuse_existing: true});
          attachedVideoIds.push(...(video.videos ?? [video]).map(v=>v.id));
          const batch = video.videos;
          const note = batch?.length ? `已登记 P${batch[0].page}–P${batch[batch.length-1].page}，共 ${batch.length} 集。` : formatVideoPageInfo(video);
          if(note) notes.push(note);
        } catch(e) {failures.push(`${request.url}：${msg(e)}`)}
      }
      if (requests.length) await refreshProject(pid);
      if(submission.cancelled){if(activeIdRef.current===pid)setNotice("已停止提交；已登记的视频保留，尚未开始整理。");return false;}
      if (failures.length) {
        if(activeIdRef.current===pid) setNotice(`以下链接未能添加，原文已保留。重试会复用已添加的视频；本轮尚未开始整理。\n${failures.join('\n')}`);
        return false;
      }
      const { message, task_id } = await api.sendMessage(pid, {
        content: mine,
        video_ids: attachedVideoIds,
        // 把用户此刻的浏览位置一起送过去。少了它，模型不知道「你说的
        // 这个 html」指哪一块，只能反问——而用户前几轮明明说过上下文。
        // 详见 backend/app/api/messages.py 的 FocusRef 注释。
        focus,
      });
      if(submission.cancelled)await api.cancelTask(task_id);
      if(activeIdRef.current!==pid)return true;
      setMessages(m=>[...m, message]);
      setLastTaskId(task_id);
      setNotice(notes.length ? notes.join('\n') : null);
      connect(task_id);

      // SSE 兜底：并行轮询等回复落库。SSE 断流时（切项目、网络抖动、
      // EventSource 被浏览器回收），回复其实早就在库里，不该只有用户
      // 那条消息孤零零挂着。用后端返回的真实 message.id 作 after_id ——
      // 不能用乐观上屏的 Date.now()，那个值比库里所有 id 都大，
      // 过滤条件会永远命中不了。
      api
        .pollForAssistantReply(pid, message.id)
        .then((reply) => {
          if (!reply || activeIdRef.current!==pid) return; // 超时：SSE 或用户刷新会兜住
          setMessages((cur) =>
            cur.some((m) => m.id === reply.id) ? cur : [...cur, reply]
          );
        })
        .catch(() => {});
      return true;
    } catch (e) {
      if(activeIdRef.current===pid)setNotice(`发送失败：${msg(e)}`);
      return false;
    } finally {
      if(pendingSend.current===submission)pendingSend.current=null;
      if(activeIdRef.current===pid)setSubmitting(false);
    }
  };

  // 中止当前这一轮。
  //
  // 刻意**不在本地乐观地隐藏进度泡**：进度泡的消失由 SSE 的 cancelled 帧
  // 驱动（useTaskEvents 里 close() + 清流式正文）。本地先隐藏的话，
  // 万一后端请求失败，界面就停在「已停」的假象上而 agent 还在跑 ——
  // 比停不下来更糟，因为用户以为已经停了。
  const stop = useCallback(async () => {
    if (pendingSend.current?.pid === activeId) {pendingSend.current.cancelled = true; return;}
    if (!lastTaskId) return;
    let r;
    try {r = await api.cancelTask(lastTaskId);}
    catch(e) {setNotice(`停止失败：${msg(e)}`);return;}
    // status 仍是 done/failed：用户慢了几秒，任务已经自己跑完了。
    // 这时**不要**动界面 —— 答案已经在库里并显示了。
    if (r && (r.status === "done" || r.status === "failed")) return;
    // 否则什么都不做，等 SSE 的 cancelled 帧自然收尾。
  }, [lastTaskId, activeId]);

  // 撤回一条消息：删掉它连同其后的回复，原文填回输入框。
  //
  // 乐观更新：先把这条及其后的消息从界面上摘掉再请求。
  // 撤回的语义是「我还没说过这句话」，界面必须立刻回到那个状态；
  // 等一次网络往返才消失会让用户以为没点上而连点，
  // 而第二次点击会打到已经被删掉的 id 上（404）。
  const recall = useCallback(
    async (messageId: number) => {
      if (activeId == null) return;
      const idx = messages.findIndex((m) => m.id === messageId);
      if (idx < 0) return;
      // 乐观切片：撤回的语义是「我还没说过这句话」，界面必须立刻回到
      // 那个状态。等一次网络往返才消失，用户会以为没点上而连点，
      // 而第二次点击会打到已被删掉的 id 上（404）。
      const doomed = messages.slice(idx);
      setMessages((cur) => cur.slice(0, idx));
      // 关掉可能还连着的 SSE：撤回之后这一轮的回复不会来了，
      // 连接留着只会每 10 秒收一个心跳帧直到后端把它关掉。
      close();
      try {
        const r = await api.recallMessage(activeId, messageId);
        // 原文填回输入框 —— 这是「撤回」与「删除」的本质区别：
        // 用户改一个字就能重发，不用重新手打。
        setDraft(r.restored);
        // 后端删的是权威结果，拉一次回来对齐
        // （乐观切片漏掉的边界情况：并发写入的新消息）。
        refreshProject(activeId).catch(() => {});
      } catch (e) {
        // 撤回失败要把消息放回去：界面上已经没有了，用户会以为这条问
        // 被删掉了（而它其实还在历史里，下一轮模型仍会读到）。
        setMessages((cur) => [...cur, ...doomed]);
        setNotice(`撤回失败：${msg(e)}`);
      }
    },
    [activeId, messages, close, refreshProject]
  );

  const [exporting, setExporting] = useState<"html" | "md" | "pdf" | null>(null);

  const onExport = async (fmt: "html" | "md" | "pdf") => {
    if (activeId == null || exporting != null) return;
    setExporting(fmt);
    try {
      if(isDesktop()) {
        const result=await api.exportLocalDoc(activeId,fmt);
        setNotice(`已导出：${result.path}`);
        await revealProject(activeId,'exports');
      } else await api.exportDoc(activeId, fmt);
    } catch (e) {
      setNotice(`导出失败：${msg(e)}`);
    } finally {
      setExporting(null);
    }
  };

  const openProject=async()=>{
    try {const chosen=await selectLocal('directory');if(!chosen)return;
      const p=await api.openLocalProject(chosen.grant);setProjects(await api.listProjects());setActiveId(p.id);
      if(activeIdRef.current===p.id)await refreshProject(p.id);
    }catch(e){setNotice(msg(e))}
  };
  const importMedia=async()=>{
    const pid=activeId;if(pid==null)return;
    try{const chosen=await selectLocal('media');if(!chosen)return;await api.importLocalMedia(pid,chosen.grant);await refreshProject(pid)}catch(e){setNotice(msg(e))}
  };
  const representedTaskIds = Array.from(new Set([
    ...(typeof task?.task_id === "string" && task.task_id ? [task.task_id] : []),
    ...messages.flatMap(message => message.activity?.steps.map(step => step.child_task_id).filter((id): id is string => typeof id === "string" && id.length > 0) ?? []),
    ...steps.map(step => step.child_task_id).filter((id): id is string => typeof id === "string" && id.length > 0),
  ]));
  return (
    <div className="app-shell" style={{'--reader-width':`${panelWidth}%`} as React.CSSProperties}>
      <Sidebar
        projects={projects}
        activeId={activeId}
        onSelect={selectProject}
        onCreate={createProject}
        onOpenSettings={() => {
          setNotice(null);
          setSettingsOpen(true);
        }}
        listError={listError}
      />
      <main className="workspace-center">
        <header className="workspace-toolbar">
          <strong>{projects.find(p=>p.id===activeId)?.title||'学习工作台'}</strong>
          {activeId != null && <button className={`workspace-view-toggle ${workspaceView === 'video' ? 'is-active' : ''}`} aria-label={workspaceView === 'chat' ? '切换到视频' : '返回会话'} aria-pressed={workspaceView === 'video'} onClick={() => {setWorkspaceView(v => v === 'chat' ? 'video' : 'chat'); setReaderOpen(true);}}><span aria-hidden="true">{workspaceView === 'chat' ? '▷' : '↩'}</span> {workspaceView === 'chat' ? '视频' : '会话'}</button>}
          {(doc||existingDocument||previewDraft||workspaceView==='video')&&<button onClick={()=>setReaderOpen(!readerOpen)} aria-label={readerOpen?'收起知识文档':'展开知识文档'}>{readerOpen?'收起文档':'知识文档'}</button>}
          {isDesktop()&&activeId!=null&&<div className="project-menu">
            <button aria-label="项目菜单" aria-expanded={menuOpen} onClick={()=>setMenuOpen(!menuOpen)}>···</button>
            {menuOpen&&<><button className="menu-dismiss" aria-label="关闭项目菜单" onClick={()=>setMenuOpen(false)}/><div className="project-menu-popover">
              <p>{storageRoot}</p>
              <button onClick={()=>{setMenuOpen(false);void revealProject(activeId).catch(e=>setNotice(msg(e)))}}>在 Finder 中打开</button>
              <button onClick={()=>{setMenuOpen(false);void revealProject(activeId,"documents").catch(e=>setNotice(msg(e)))}}>查看知识文件</button>
              <button onClick={()=>{setMenuOpen(false);void openProject()}}>重新定位项目目录…</button>
            </div></>}
          </div>}
        </header>
        {storageStatus&&storageStatus.status!=="saved"&&isDesktop()&&<div className={`storage-status storage-status--${storageStatus.status}`}>
          {storageStatus.message}
          {['error','unavailable'].includes(storageStatus.status)&&<><button onClick={async()=>{if(activeId!=null)setStorageStatus(await api.saveLocalProject(activeId))}}>重试保存</button><button onClick={openProject}>重新定位目录</button></>}
        </div>}
      <div className="workspace-conversation" hidden={workspaceView !== "chat"}>
      <ChatPanel
        key={activeId}
        timelineManaged
        visible={workspaceView === "chat"}
        modelName={modelName}
        modelSettingsStatus={modelSettingsStatus}
        onModelSettings={()=>setSettingsOpen(true)}
        onImportMedia={isDesktop()?importMedia:undefined}
        messages={messages}
        videos={videos}
        activeProjectId={activeId}
        onSend={send}
        task={task}
        taskError={taskError}
        stale={stale}
        notice={notice}
        onDismissNotice={() => setNotice(null)}
        onRefresh={() => {
          setNotice(null);
          if (activeId != null) {
            refreshProject(activeId).catch((e) =>
              setNotice(`刷新失败：${msg(e)}`)
            );
          }
        }}
        steps={steps}
        streamText={streamText}
        submitting={submitting}
        onStop={stop}
        onRecall={recall}
        draft={draft}
        onDraftChange={setDraft}
      ><ProjectTasks projectId={activeId} presentation="conversation" representedTaskIds={representedTaskIds} messages={messages} videosPresent={videos.length > 0} onRecall={recall} liveTask={task} liveSteps={steps} onRefresh={()=>{if(activeId!=null)refreshProject(activeId).catch(()=>{})}} onDraft={d=>{setPreviewDraft(d);setReaderOpen(true)}}/></ChatPanel>
      </div>
      {workspaceView === "video" && activeId != null && <VideoPlayer projectId={activeId} videos={videos} selectedId={selectedVideoId} onSelect={setSelectedVideoId} onProcess={videoId => setProcessingVideo({projectId:activeId,videoId})} />}
      </main>
      {readerOpen&&(doc||existingDocument||previewDraft||workspaceView==='video')&&<>
      <div className="reader-divider" role="separator" aria-label="调整阅读区宽度" aria-orientation="vertical" tabIndex={0}
        onKeyDown={e=>{if(e.key==='ArrowLeft')setPanelWidth(w=>Math.min(65,w+2));if(e.key==='ArrowRight')setPanelWidth(w=>Math.max(30,w-2));}}
        onPointerDown={e=>{const start=e.clientX,width=panelWidth;e.currentTarget.setPointerCapture(e.pointerId);
          const move=(ev:PointerEvent)=>setPanelWidth(Math.max(30,Math.min(65,width+(start-ev.clientX)/window.innerWidth*100)));
          const end=()=>{window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',end)};
          window.addEventListener('pointermove',move);window.addEventListener('pointerup',end,{once:true});}} />
      <section className="workspace-reader">
      {previewDraft&&<div className="draft-banner">待校验草稿 · 尚未合并至正式文档<button onClick={()=>setPreviewDraft(null)}>返回正式文档</button></div>}
      {existingDocument&&!previewDraft&&<div className="document-tabs"><button className={!doc||showOriginal?'is-active':''} onClick={()=>setShowOriginal(true)}>原有 knowledge.html</button>{doc&&<button className={!showOriginal?'is-active':''} onClick={()=>setShowOriginal(false)}>整理结果</button>}</div>}
      {existingDocument&&!previewDraft&&(!doc||showOriginal)?<><p className="original-document-note">原文件只读预览 · 新整理结果另存，不会覆盖原件</p><iframe className="original-document" title="原有知识文档" sandbox="" src={`/api/projects/${activeId}/existing-document`}/></>:<KnowledgePanel
        doc={previewDraft??doc}
        videosPresent={videos.length > 0}
        taskStatus={knowledgeTaskStatus}
        onExport={previewDraft?undefined:onExport}
        exporting={exporting}
        onFocusChange={(f) => {
          focusRef.current = f;
        }}
      />}
      </section></>}
      {processingVideo && <ProcessingDialog projectId={processingVideo.projectId} onClose={()=>setProcessingVideo(null)} onSubmit={process} />}
      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onNotice={setNotice}
      />
    </div>
  );
}

function msg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
