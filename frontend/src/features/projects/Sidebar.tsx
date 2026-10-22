import { useState } from "react";
import {isDesktop,selectLocal,type Selection} from "../../api/desktop";
import type { Project } from "../../api/types";

export function Sidebar({
  projects,
  activeId,
  onSelect,
  onCreate,
  onOpenSettings,
  listError,
}: {
  projects: Project[];
  activeId: number | null;
  onSelect: (id: number) => void;
  onCreate: (title: string, subject: string, directoryGrant?: string) => void | Promise<void>;
  // 可选：没有这个 prop 就不渲染齿轮（Sidebar 自己的测试不传它）。
  // 不设成必需是为了让组件保持可独立测试，而不是每个测试都得编一个空函数。
  onOpenSettings?: () => void;
  listError?: string | null;
}) {
  // 内联新建表单：比 window.prompt 更符合 macOS 风格，也不阻塞主线程。
  const [creating, setCreating] = useState(false);
  const [title, setTitle] = useState("");
  const [subject, setSubject] = useState("");

  const [selection,setSelection]=useState<Selection|null>(null);
  const [error,setError]=useState('');
  const [busy,setBusy]=useState(false);
  const desktop=isDesktop();
  const pick=async()=>{try{const chosen=await selectLocal('directory');if(chosen){setSelection(chosen);if(!title.trim())setTitle(chosen.name);}}catch(e){setError(String(e))}};
  const submit = async () => {
    const t=title.trim();if(!t||busy||(desktop&&!selection))return;
    setBusy(true);setError('');
    try {
      if(desktop)await onCreate(t,subject.trim(),selection!.grant);
      else await onCreate(t,subject.trim());
      setTitle('');setSubject('');setSelection(null);setCreating(false);
    }catch(e){setError(e instanceof Error?e.message:String(e));setSelection(null);}
    finally{setBusy(false)}
  };

  return (
    <aside className="sidebar">
      <div className="sidebar__header">
        <span className="sidebar__title">学习工作台</span>
        {/* 齿轮放在标题栏右侧：后端设置是「整个应用级」的配置，
            挂在某个项目下面会让人以为它只对这个项目生效。 */}
        {onOpenSettings && (
          <button
            className="sidebar__settings"
            onClick={onOpenSettings}
            title="后端设置（模型、密钥、预算）"
            aria-label="后端设置"
          >
            ⚙
          </button>
        )}
      </div>

      {creating ? (
        <div className="new-project-form">
          <input
            className="new-project-form__field"
            placeholder="项目名称"
            value={title}
            autoFocus
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
              if (e.key === "Escape") setCreating(false);
            }}
          />
          <input
            className="new-project-form__field"
            placeholder="学科 / 领域（可选）"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
              if (e.key === "Escape") setCreating(false);
            }}
          />
          {desktop&&<><button className="btn-new" onClick={pick} disabled={busy}>选择保存目录</button><p className="directory-label">{selection?.path||'视频与知识文档将保存在此目录'}{selection?.existing_project?' · 已有项目，将恢复内容':selection?.existing_document?' · 已发现 knowledge.html，将保留并展示':''}</p></>}
          {error&&<p role="alert" className="form-error">{error}</p>}
          <div className="new-project-form__actions">
            <button
              className="btn-new btn-new--primary"
              onClick={submit}
              disabled={!title.trim()||busy||(desktop&&!selection)}
            >
              {selection?.existing_project ? "继续此项目" : "创建"}
            </button>
            <button className="btn-new" onClick={() => setCreating(false)}>
              取消
            </button>
          </div>
        </div>
      ) : (
        <button className="btn-new" onClick={() => setCreating(true)}>
          + 新建项目
        </button>
      )}

      <div className="sidebar-section-label">知识项目</div>
      <div className="project-list">
        {listError ? (
          <div className="hint-bar hint-bar--warn" style={{ margin: 8 }}>
            {listError}
          </div>
        ) : (
          projects.map((p) => (
            <button
              key={p.id}
              className={`project-item ${p.id === activeId ? "project-item--active" : ""}`}
              onClick={() => onSelect(p.id)}
            >
              <span className="project-item__title">{p.title}</span>
              {p.subject&&<span className="project-item__meta">{p.subject}</span>}
            </button>
          ))
        )}
        {projects.length === 0 && !creating && !listError && (
          <div className="empty-state" style={{ padding: 16 }}>
            <div>还没有项目</div>
            <div style={{ fontSize: 12 }}>点击上方「+ 新建项目」开始</div>
          </div>
        )}
      </div>
    </aside>
  );
}
