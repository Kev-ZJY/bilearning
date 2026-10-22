export type Selection = {grant:string;path:string;name:string;existing_project?:boolean;existing_document?:boolean};
type NativeBridge={postMessage:(message:Record<string,unknown>)=>Promise<unknown>};
declare global {interface Window {webkit?:{messageHandlers?:{desktop?:NativeBridge}}}}
export const isDesktop=()=>!!window.webkit?.messageHandlers?.desktop;
export async function selectLocal(kind:'directory'|'media'):Promise<Selection|null> {
  const bridge=window.webkit?.messageHandlers?.desktop;
  if(!bridge) throw new Error('请从 macOS 桌面应用选择本机目录');
  return await bridge.postMessage({action:kind}) as Selection|null;
}
export async function revealProject(projectId:number,area="project") {
  await window.webkit?.messageHandlers?.desktop?.postMessage({action:'reveal',projectId,area});
}
