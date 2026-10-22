/** Extract complete links without treating the surrounding request as a URL. */
export function videoLinks(text: string): string[] {
  return [...new Set((text.match(/https?:\/\/[^\s<>"'，。；、（）()「」『』“”‘’\[\]]+|\bBV[a-zA-Z0-9]{10}(?:\?p=\d+)?/g) ?? [])
    .map(url => url.replace(/[.,;!?]+$/, "")))];
}


/** An explicit range belongs to one series; ambiguous multi-series requests fail before import. */
export function videoRequests(text: string): {url:string; page_end?:number}[] {
  const links = videoLinks(text).filter(url => /^(?:BV[a-zA-Z0-9]{10}|https?:\/\/(?:www\.)?bilibili\.com\/video\/BV)/.test(url));
  const ranges = [...text.matchAll(/(?:从\s*)?[pP]\s*=?\s*(\d+)\s*(?:到|至|[-~～—–])\s*[pP]?\s*=?\s*(\d+)/g)];
  if (!ranges.length) return links.map(url=>({url}));
  if (ranges.length !== 1 || links.length !== 1) throw new Error("请每次为一个视频合集指定分 P 范围。");
  const start=Number(ranges[0][1]), end=Number(ranges[0][2]);
  if(start<1 || end<start || end-start>=100) throw new Error("分 P 范围无效，请按起点到终点填写（每次最多 100 P）。");
  const url=new URL(links[0].startsWith('BV') ? `https://www.bilibili.com/video/${links[0]}` : links[0]);
  const linkedPage=url.searchParams.get('p');
  if(linkedPage !== null && Number(linkedPage)!==start) throw new Error("链接中的分 P 与范围起点不一致，请明确要整理的范围。");
  url.searchParams.set('p', String(start));
  return [{url:url.toString(),page_end:end}];
}
