import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import type { Cheatsheet, Example, FocusRef, IndependentReview, KnowledgeDoc, KnowledgeFigure, Point } from "../../api/types";

// 掌握度目前没有后端端点，先在浏览器本地持久化。
// key 用「项目文档标题 + 知识点标题」而不是 LLM 生成的 point.id：
// id 在多视频融合时会重名（fusion 里 m1/p1 满天飞），标题稳定得多。
const MASTERY_KEY = "vlwt.mastery.v1";

const STATES = ["new", "learning", "mastered"] as const;
const STATE_LABELS: Record<string, string> = {
  new: "未学",
  learning: "学习中",
  mastered: "已掌握",
};

function safeFigure(figure: KnowledgeFigure): boolean {
  return figure.kind === "video_frame" && Number.isInteger(figure.second) && figure.second >= 0
    && figure.source_ref === `frame:${figure.second}` && typeof figure.src === "string"
    && figure.src.length <= 136559 && /^data:image\/jpeg;base64,\/9j\/[A-Za-z0-9+/]*={0,2}$/.test(figure.src);
}

function decisionNodes(example: Example): string[] {
  if (example.nodes?.length) return example.nodes;
  return (example.content ?? "").split(/\n|→/).map(line => line.replace(/^\s*(?:[-*]|\d+[.)、])\s*/, "").trim()).filter(Boolean);
}

function safeDiagram(example: Example): boolean {
  return example.type === "diagram" && ["parallel", "sequence"].includes(example.layout ?? "")
    && Array.isArray(example.nodes) && example.nodes.length >= 2 && example.nodes.length <= 8
    && example.nodes.every(node => typeof node === "string" && node.trim().length > 0 && node.length <= 120)
    && typeof example.content === "string" && example.content.trim().length > 0 && example.content.length <= 1200
    && example.basis === "illustrative" && example.label === "解释性例子（非来源原例）";
}

function reviewNotes(review: IndependentReview) {
  const label: Record<string, string> = { reviewed: "已审查", revised: "已审查并修改", stale: "文档已更新，原审查报告已过期" };
  const actions: Record<string, string> = { accept: "采纳", reject: "未采纳", defer: "待核实" };
  const sources = (review.web_sources ?? []).filter(source => {
    try { return ["http:", "https:"].includes(new URL(source.url).protocol); } catch { return false; }
  });
  return <details className="kd-review-notes">
    <summary>核验说明 · {label[review.status] ?? "状态待确认"}</summary>
    {review.scope && <p>审查范围：{review.scope}</p>}
    {review.limitations?.map((text, i) => <p className="kd-review-limitation" key={`limitation-${i}`}>待核实：{text}</p>)}
    {review.findings?.map((finding, i) => <p key={`finding-${i}`}>{finding.issue}{finding.suggestion && <> · 建议：{finding.suggestion}</>}</p>)}
    {review.decisions?.map((decision, i) => <p key={`decision-${i}`}>{actions[decision.decision] ?? "处理说明"}：{decision.reason}</p>)}
    {sources.length > 0 && <><p>参考来源</p><ul>{sources.map((source, i) => <li key={`source-${i}`}><a href={source.url} target="_blank" rel="noopener noreferrer">{source.title || source.url}</a></li>)}</ul></>}
  </details>;
}

function loadMastery(): Record<string, string> {
  try {
    const raw = localStorage.getItem(MASTERY_KEY);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    // localStorage 不可用（隐私模式 / 配额满）时退化为「全都不标记」，不影响浏览
    return {};
  }
}

function masteryKey(docKey: string, point: Point): string {
  return `${docKey}::${point.title}`;
}

function MasteryToggle({
  point,
  docKey,
  onToggle,
}: {
  point: Point;
  docKey: string;
  onToggle: (key: string, next: string) => void;
}) {
  const [map, setMap] = useState<Record<string, string>>(() => loadMastery());
  const key = masteryKey(docKey, point);
  const cur = map[key] ?? "new";
  const idx = STATES.indexOf(cur as (typeof STATES)[number]);

  return (
    <div
      className="mastery-toggle"
      onClick={(e) => e.stopPropagation()}
      role="group"
      aria-label={`掌握度：${STATE_LABELS[cur] ?? cur}`}
    >
      {STATES.map((s, i) => (
        <button
          key={s}
          className={`mastery-dot ${idx >= 0 && i <= idx ? "mastery-dot--active" : ""}`}
          title={STATE_LABELS[s]}
          aria-label={STATE_LABELS[s]}
          onClick={() => {
            // 再点一次当前档位就回退到「未学」，否则用户没法取消标记
            const next = cur === s ? "new" : s;
            setMap((m) => {
              const nm = { ...m, [key]: next };
              try {
                localStorage.setItem(MASTERY_KEY, JSON.stringify(nm));
              } catch {
                // 写不进去就在内存里生效，不打断交互
              }
              return nm;
            });
            onToggle(key, next);
          }}
        />
      ))}
    </div>
  );
}

function CheatsheetTable({ sheet }: { sheet: Cheatsheet }) {
  const headers = sheet.headers ?? [];
  const rows = sheet.rows ?? [];
  if (headers.length === 0 && rows.length === 0) return null;
  return (
    <div className="kd-cheatsheet">
      {sheet.title && <h3 className="kd-cheatsheet__title">{sheet.title}</h3>}
      <table className="kd-table">
        {headers.length > 0 && (
          <thead>
            <tr>
              {headers.map((h, hi) => (
                <th key={hi}>{h}</th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}>
              {r.map((c, ci) => (
                <td key={ci}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function KnowledgePanel({
  doc,
  onExport,
  exporting,
  onFocusChange,
  videosPresent = false,
  taskStatus = null,
}: {
  doc: KnowledgeDoc | null;
  videosPresent?: boolean;
  taskStatus?: string | null;
  onExport?: (fmt: "html" | "md" | "pdf") => void;
  // 收窄成联合类型而不是 string：之前这里是 string，
  // 结果 `busy === f`（boolean vs string）静默恒为 false，
  // 「导出中…」永远不显示，只有 tsc 能发现。
  exporting?: "html" | "md" | "pdf" | null;
  /**
   * 上报「用户此刻在看哪个知识点」。
   *
   * **这是修「agent 反复反问已知信息」的关键一环。** 用户报的真实症状：
   * 他说「把 html 改成 Apple 风格」，助手反问「是苹果公司、苹果产品、
   * 苹果网站应用，还是一个网页」——而后端完全不知道他正对着哪一块。
   * 模型拿不到信息时反问是合理的，它不是笨，是架构没把信息送到。
   *
   * 上报的时机是**点击知识点标题**（用户明确的意图信号），
   * 而不是滚动位置。滚动上报要装 IntersectionObserver 且噪声极大
   * （快速划过会连发几十次），而点击是零成本、零歧义的。
   * 副作用是「看一眼但没点」不会被上报——那种情况用户如果接着提问，
   * 问的通常就是整个文档，后端不写焦点段也不会出错。
   *
   * 可选：没传就不上报，App 侧对应地不发送 focus。
   */
  onFocusChange?: (focus: FocusRef | null) => void;
}) {
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  // 目录跳转用 ref 而不是 document.getElementById：
  // m.id 来自 LLM，不保证唯一（多模块文档里 "m1" 极常见），
  // getElementById 只会返回第一个，目录会跳到错误的模块。
  const moduleRefs = useRef(new Map<string, HTMLDivElement>());

  const docKey = doc?.meta?.title ?? "";

  // 换文档时清掉折叠态：id 会重名，上个项目的折叠不该带到新项目
  useEffect(() => {
    setCollapsed(new Set());
    moduleRefs.current.clear();
  }, [doc]);

  // 换文档时把焦点清成 null：焦点里的 doc_title 属于上一篇文档，
  // 留着会让助手以为用户还在看上一篇的内容。
  useEffect(() => {
    onFocusChange?.(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docKey]);

  const filtered = useMemo(() => {
    if (!doc) return null;
    const q = query.trim().toLowerCase();
    return {
      ...doc,
      modules: (doc.modules ?? [])
        .map((m, mi) => ({
          ...m,
          // 字段可能是 undefined/null（LLM 输出不保证齐全），
          // 直接拼字符串会产出 "undefined"/"null" 参与匹配，
          // 搜这两个词会捞出一堆无关结果。用 ?? "" 兜底。
          points: q
            ? (m.points ?? []).filter((p) =>
                `${p.title ?? ""} ${p.anchor ?? ""} ${p.body ?? ""}`
                  .toLowerCase()
                  .includes(q)
              )
            : (m.points ?? []),
          // 用位置而非 id 做 ref key，重名时也不会互相覆盖。
          // 必须无条件注入：早前这里只在有搜索词时才加 _key，
          // 于是「不搜索」这个最常见的状态下 key 退回 m.id，
          // 重名模块撞 duplicate key，折叠态互相串。
          _key: `${mi}`,
        }))
        .filter((m) => (q ? m.points.length > 0 : true)),
    };
  }, [doc, query]);

  if (!filtered || !doc?.modules?.length) {
    const emptyMessage = !videosPresent
      ? "先添加视频并点击「整理」，助手会生成可交互知识点"
      : taskStatus === "running"
        ? "正在整理，知识点完成后会显示在这里。"
        : taskStatus === "failed"
          ? "整理未完成，查看会话中的失败记录。"
          : "整理后，知识点会显示在这里。";
    return (
      <section className="knowledge-panel">
        <div className="empty-state">
          <div>知识点将在这里呈现</div>
          <div style={{ fontSize: 12 }}>
            {emptyMessage}
          </div>
        </div>
      </section>
    );
  }

  const keyOf = (m: { id: string; _key?: string }) => m._key ?? m.id;

  const toggleModule = (key: string) => {
    const next = new Set(collapsed);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setCollapsed(next);
  };

  const busy = exporting != null || !onExport;

  return (
    <section className="knowledge-panel">
      <header className="knowledge-panel__header">
        <input
          className="knowledge-panel__search"
          placeholder="搜索知识点…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {(["html", "md", "pdf"] as const).map((f) => (
          <button
            key={f}
            className="btn-new"
            // PDF 转换在后端要走子进程，可能耗时数秒；
            // 不置灰的话用户能无限连点，堆出多个并发转换任务。
            disabled={busy}
            onClick={() => onExport?.(f)}
          >
            {exporting === f ? "导出中…" : f.toUpperCase()}
          </button>
        ))}
      </header>

      <article className="knowledge-doc">
        {filtered.meta?.visual_review && <p role="status" className="settings__hint">最近一次整理的画面核对范围：{filtered.meta.visual_review.reason || "未提供核对范围"}</p>}
        {filtered.modules.length === 0 && <p>没有匹配的知识点</p>}
        <h1 className="kd-title">{filtered.meta?.title}</h1>
        {filtered.meta?.positioning && (
          <p className="kd-positioning">{filtered.meta.positioning}</p>
        )}
        {filtered.meta?.independent_review && reviewNotes(filtered.meta.independent_review)}

        <nav className="kd-toc">
          {filtered.modules.map((m) => {
            const k = keyOf(m as { id: string; _key?: string });
            return (
              <div
                key={k}
                className="kd-toc__item"
                onClick={() =>
                  moduleRefs.current
                    .get(k)
                    ?.scrollIntoView({ behavior: "smooth", block: "start" })
                }
              >
                {m.title}
              </div>
            );
          })}
        </nav>

        {filtered.modules.map((m) => {
          const k = keyOf(m as { id: string; _key?: string });
          return (
            <div
              key={k}
              ref={(el) => {
                if (el) moduleRefs.current.set(k, el);
                else moduleRefs.current.delete(k);
              }}
              className={`kd-module ${collapsed.has(k) ? "kd-module--collapsed" : ""}`}
            >
              <div className="kd-module__header" onClick={() => toggleModule(k)}>
                <span className="kd-module__chevron">▼</span>
                <h2 className="kd-module__title">{m.title}</h2>
              </div>
              <div className="kd-module__body">
                {m.points.map((p) => (
                  <div key={p.id} className="kd-point">
                    <div
                      className="kd-point__header"
                      // 点击知识点 = 用户对「我现在看的是哪一块」表态。
                      // 放在 header（而非 title 标签）上，是为了让掌握度
                      // 圆点的点击不冒泡上来 —— MasteryToggle 自己 stopPropagation，
                      // 但那是在它内部；打在整个 header 上更符合直觉。
                      onClick={() =>
                        onFocusChange?.({
                          doc_title: docKey,
                          point_title: p.title ?? "",
                          module_title: m.title ?? "",
                          view: "html",
                        })
                      }
                    >
                      <h3 className="kd-point__title">{p.title}</h3>
                      <MasteryToggle
                        point={p}
                        docKey={docKey}
                        onToggle={() => {}}
                      />
                    </div>
                    {p.anchor && <p className="kd-point__anchor">{p.anchor}</p>}
                    {p.body && <p className="kd-point__body">{p.body}</p>}
                    {p.figures?.map((figure, i) => safeFigure(figure) ? (
                      <figure className="kd-figure" key={`${figure.id}-${i}`}>
                        <img src={figure.src} alt={figure.alt} loading="lazy" />
                        <figcaption>{figure.caption} · 来源 {figure.second} 秒</figcaption>
                      </figure>
                    ) : <p className="kd-warning" key={`invalid-figure-${i}`}>插图数据无效，请重新整理该知识点。</p>)}
                    {p.examples?.map((ex, i) =>
                      <div key={i}>
                      {ex.label && ex.type !== "diagram" && <p className="kd-example-label">{ex.label}</p>}
                      {ex.type === "diagram" ? (safeDiagram(ex) ? (
                        <figure className={`kd-diagram ${ex.layout}`}>
                          <figcaption>解释性示意图（非视频原图）</figcaption>
                          <div className="kd-diagram-nodes">{ex.nodes!.map((node, ni) => <Fragment key={ni}>
                            {ni > 0 && ex.layout === "sequence" && <span className="kd-diagram-arrow" aria-hidden="true">→</span>}
                            <div className="kd-diagram-node">{node}</div>
                          </Fragment>)}</div>
                          <p>{ex.content}</p>
                        </figure>
                      ) : <p className="kd-warning">示意图数据无效，请重新整理该知识点。</p>) : ex.type === "table" && ex.headers && ex.rows ? (
                        <table className="kd-table">
                          <thead>
                            <tr>
                              {ex.headers.map((h, hi) => (
                                <th key={hi}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {ex.rows.map((r, ri) => (
                              <tr key={ri}>
                                {r.map((c, ci) => (
                                  <td key={ci}>{c}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : ex.type === "decision_tree" ? (
                        <ol className="kd-decision-tree">{decisionNodes(ex).map((node, ni) => <li key={ni}>{node}</li>)}</ol>
                      ) : (
                        <div className="kd-point__example">
                          {ex.content}
                        </div>
                      )}</div>
                    )}
                    {p.warnings?.map((w, i) => (
                      <div key={i} className="kd-warning">
                        ⚠ {w}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          );
        })}

        {/* 速查表：后端认真生成、PDF 里也渲染了，界面上不能不显示 */}
        {filtered.appendix?.cheatsheets?.length ? (
          <section className="kd-appendix">
            <h2 className="kd-module__title">附录 · 速查表</h2>
            {filtered.appendix.cheatsheets.map((cs, i) => (
              <CheatsheetTable key={i} sheet={cs} />
            ))}
          </section>
        ) : null}
      </article>
    </section>
  );
}
