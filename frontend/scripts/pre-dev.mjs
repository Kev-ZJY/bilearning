// dev 启动前置检查：端口被占时提前提醒。
//
// 为什么需要：端口冲突有两种失败方式。
//   1) 静默顺延——vite 抢不到端口时自己换到 5174/5181…，你按记忆里的旧
//      端口打开浏览器，看到的是占用该端口的**别的服务**，很容易误判成
//      「这个项目坏了」。
//   2) 启动失败——strictPort:true 时 vite 直接退出，报 EADDRINUSE。
//      这个反而是好的，冲突立刻可见，不会变成诡异的假象。
//
// 探测必须同时试 IPv4 和 IPv6：本机 5173 上曾同时存在
// 127.0.0.1:5173（某个长期运行的本地服务）和 [::1]:5173（vite），
// 只探 127.0.0.1 会漏报，导致 vite 启动失败却在预检里显示"端口可用"。
import { createConnection } from 'node:net';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

// 从 vite.config.ts 里读端口和 strictPort，避免这里写死一个和 vite 不一致的值。
// 用正则而非 import：直接 import 配置会把 vite 插件整套拖进来。
function readServerConfig() {
  try {
    const src = readFileSync(resolve(here, '../vite.config.ts'), 'utf8');
    const block = src.match(/server\s*:\s*\{/s);
    const port = src.match(/server\s*:\s*\{[^}]*?\bport\s*:\s*(\d+)/s);
    const strict = src.match(/server\s*:\s*\{[^}]*?\bstrictPort\s*:\s*(\w+)/s);
    return {
      port: port ? Number(port[1]) : 5180,
      strictPort: strict ? strict[1] === 'true' : true,
    };
  } catch {
    return { port: 5180, strictPort: true };
  }
}

const { port, strictPort } = readServerConfig();

// 同时探两个地址族：绑定在 [::1] 的服务用 127.0.0.1 探不到，反之亦然。
const HOSTS = ['127.0.0.1', '::1'];

function isPortAlive(host, p) {
  return new Promise((resolveFn) => {
    const sock = createConnection({ host, port: p });
    sock.on('connect', () => { sock.destroy(); resolveFn(true); });
    sock.on('error', () => resolveFn(false));
    // 兜底：万一连接既不成功也不报错（比如被防火墙静默丢弃）
    setTimeout(() => { sock.destroy(); resolveFn(false); }, 700);
  });
}

const busy = (await Promise.all(HOSTS.map((h) => isPortAlive(h, port))))
  .map((alive, i) => (alive ? HOSTS[i] : null))
  .filter(Boolean);

if (busy.length > 0) {
  if (strictPort) {
    console.error(`\n✖  端口 ${port} 已被占用（${busy.join('、')}），vite 会直接启动失败。`);
    console.error(`   先关掉占用进程，或把 vite.config.ts 的 port 改成别的空闲端口。\n`);
  } else {
    console.log(`\n⚠️  端口 ${port} 已被占用（${busy.join('、')}），Vite 会自动顺延。`);
    console.log(`   请认下面 Vite 打印的 Local 地址，别打开 ${port}。\n`);
  }
  process.exit(strictPort ? 1 : 0);
}
