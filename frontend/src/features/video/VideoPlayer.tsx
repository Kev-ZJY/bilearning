import { useEffect, useState } from "react";
import { getPlayback } from "../../api/client";
import type { Playback, Video } from "../../api/types";

function validPlayback(playback: Playback, projectId: number, videoId: number): boolean {
  if (playback.kind === "unavailable") return playback.url === null
    && ["missing", "audio_only", "video_only", "unmerged", "invalid"].includes(playback.cache_status || "");
  if (playback.kind === "embed") {
    try {
      const url = new URL(playback.url);
      return url.origin === "https://player.bilibili.com" && url.pathname === "/player.html";
    } catch { return false; }
  }
  return ["video", "audio"].includes(playback.kind)
    && playback.url === `/api/projects/${projectId}/videos/${videoId}/media`;
}

export function VideoPlayer({ projectId, videos, selectedId, onSelect, onProcess }: {
  projectId: number;
  videos: Video[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  onProcess?: (videoId: number) => void;
}) {
  const video = videos.find(v => v.id === selectedId) ?? videos[0];
  const [playback, setPlayback] = useState<Playback | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    let live = true;
    setPlayback(null); setError("");
    if (!video) {setLoading(false); return;}
    setLoading(true);
    getPlayback(projectId, video.id).then(result => {
      if (!live) return;
      if (!validPlayback(result, projectId, video.id)) throw new Error("播放地址无效，请重新加载来源");
      setPlayback(result);
    }).catch(e => {
      if (live) setError(e instanceof Error ? e.message : "无法加载视频");
    }).finally(() => {if (live) setLoading(false);});
    return () => {live = false;};
  }, [projectId, video?.id, retry]);

  if (!video) return <section className="video-player-panel"><div className="player-empty"><h2>还没有视频来源</h2><p>返回会话，粘贴视频链接或添加本地媒体。</p></div></section>;
  const mediaError = () => setError("媒体加载失败，请重试或打开原视频");
  const bilibili = video.source === "bilibili" || !!video.bvid;
  const canProcess = onProcess && ["pending", "failed", "cancelled", "done"].includes(video.status);
  const processLabel = video.status === "pending" ? "整理" : "重新整理";
  const verifiedSource = playback?.source_url?.startsWith("https://www.bilibili.com/video/") ? playback.source_url : null;
  const source = verifiedSource ?? (video.bvid && /^BV[0-9A-Za-z]+$/.test(video.bvid) ? `https://www.bilibili.com/video/${video.bvid}/` : null);
  return <section className="video-player-panel" aria-label="视频学习区">
    <div className="player-source-picker">
      <label htmlFor="player-video">视频来源</label>
      <select id="player-video" aria-label="选择视频" value={video.id} onChange={e => onSelect(Number(e.target.value))}>
        {videos.map(v => <option value={v.id} key={v.id}>{v.title || v.bvid || `视频 ${v.id}`}</option>)}
      </select>
    </div>
    <div className="player-heading">
      {playback?.page && <span>第 {playback.page} 集</span>}
      {canProcess && <button type="button" aria-label={`${processLabel} ${video.title || video.bvid || "视频"}`} onClick={() => onProcess(video.id)}>{processLabel}</button>}
    </div>
    <div className="video-stage">
      {loading && <p role="status">正在加载视频来源…</p>}
      {error ? <div className="player-error"><p role="alert">{error}</p><button onClick={() => setRetry(n => n + 1)}>重试加载</button></div> : playback && (
        playback.kind === "unavailable" ? <div className="player-empty" role="status"><h3>本地视频尚未就绪</h3><p>{playback.cache_message}</p><button onClick={() => setRetry(n => n + 1)}>刷新播放状态</button></div> :
        playback.kind === "embed" ? <iframe key={playback.url} title="视频播放器" src={playback.url} sandbox="allow-scripts allow-same-origin allow-presentation" allow="fullscreen; picture-in-picture" allowFullScreen referrerPolicy="no-referrer" onError={mediaError} /> :
        playback.kind === "audio" ? <audio key={playback.url} aria-label="音频播放器" controls preload="metadata" src={playback.url} onError={mediaError} /> :
        <video key={playback.url} aria-label="视频播放器" controls playsInline preload="metadata" src={playback.url} onError={mediaError} />
      )}
    </div>
    <footer className="player-footer" aria-label="视频来源说明"><span>{playback?.storage === "cache" && playback.cache_status === "ready" ? "来源：本地缓存" : bilibili ? "来源：B站" : "来源：本地媒体"}</span>{source && <a href={source} target="_blank" rel="noopener noreferrer">{verifiedSource ? "在 B站打开" : "在 B站查看合集"}</a>}</footer>
  </section>;
}
