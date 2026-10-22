import { useEffect, useRef, useState } from 'react';
import type { ProcessingOptions } from '../../api/types';

const defaults: ProcessingOptions = { document_mode: null, example_strategy: null, integration: 'merge', hard_subtitles: false, subtitle_crop_bottom: .3 };

export function ProcessingDialog({projectId=0,onClose,onSubmit}:{projectId?:number;onClose:()=>void;onSubmit:(intent:string, options:ProcessingOptions)=>Promise<void>}) {
  const key=`vlwt.processing.${projectId}`;
  const [saved]=useState(()=>{try {return JSON.parse(localStorage.getItem(key)||'{}')}catch{return {}}});
  const [options,setOptions]=useState<ProcessingOptions>({...defaults,...saved.options});
  const [intent,setIntent]=useState<string>(saved.intent||'');
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  const dialog=useRef<HTMLDivElement>(null);
  useEffect(()=>{
    const previous=document.activeElement as HTMLElement | null;
    dialog.current?.querySelector('select')?.focus();
    return ()=>previous?.focus();
  },[]);
  return <div className="modal-mask"><div ref={dialog} className="modal" role="dialog" aria-modal="true" aria-label="整理要求" onKeyDown={e=>{
    if(e.key==='Escape'&&!busy) onClose();
    if(e.key==='Tab') {
      const elements=dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input,select,textarea');
      if(!elements?.length)return;
      const first=elements[0], last=elements[elements.length-1];
      if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus()}
      else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus()}
    }
  }}>
    <header className="modal__header"><h2>整理要求</h2><button className="modal__close" disabled={busy} onClick={onClose} aria-label="关闭">×</button></header>
    <form className="modal__body" onSubmit={async e=>{
      e.preventDefault();setBusy(true);setError('');
      try {
        await onSubmit(intent||'整理视频知识点', options);
        try {localStorage.setItem(key,JSON.stringify({intent,options}))} catch { /* 保存偏好失败不影响任务 */ }
        onClose();
      } catch(error) {setError(error instanceof Error?error.message:String(error))}
      finally {setBusy(false)}
    }}>
      <label className="settings__field">文档定位<select value={options.document_mode||''} onChange={e=>setOptions({...options,document_mode:(e.target.value||null) as ProcessingOptions['document_mode']})}><option value="">沿用补充要求（默认背诵清单）</option><option value="recitation">背诵清单</option><option value="textbook">体系化教材</option></select></label>
      <label className="settings__field">整合方式<select value={options.integration} onChange={e=>setOptions({...options,integration:e.target.value as ProcessingOptions['integration']})}><option value="merge">融合进当前文档</option><option value="replace">重写替换（保留旧版本）</option></select></label>
      <label className="settings__field">例子策略<select value={options.example_strategy||''} onChange={e=>setOptions({...options,example_strategy:(e.target.value||null) as ProcessingOptions['example_strategy']})}><option value="">沿用补充要求（默认按需给）</option><option value="as_needed">按需给例子</option><option value="course">贴合课程实例</option><option value="analogy">使用日常类比</option></select></label>
      <label className="settings__field">补充要求<textarea value={intent} onChange={e=>setIntent(e.target.value)} placeholder="例如：按因果关系组织，保留关键代码"/></label>
      <label><input type="checkbox" checked={options.hard_subtitles} onChange={e=>setOptions({...options,hard_subtitles:e.target.checked})}/>视频带有硬字幕，尝试逐秒识别</label>
      {options.hard_subtitles&&<label className="settings__field">底部字幕区域占画面比例<input type="number" min="0.01" max="1" step="0.01" required value={options.subtitle_crop_bottom} onChange={e=>setOptions({...options,subtitle_crop_bottom:Number(e.target.value)})}/></label>}
      <p className="settings__hint">优先读取可用字幕。硬字幕识别在本机运行，耗时随视频长度与设备变化；识别不足会回退转写。画面核对范围会在结果中说明。</p>
      {error&&<p role="alert" className="settings__error">{error}</p>}
      <div className="modal__actions"><button type="button" className="btn-new" disabled={busy} onClick={onClose}>取消</button><button className="btn-new btn-new--primary" disabled={busy}>{busy?'提交中…':'开始整理'}</button></div>
    </form>
  </div></div>;
}
