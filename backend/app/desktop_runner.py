"""Native app entry. Never loads .env or inherits developer model credentials."""

import os
import signal
import socket
import threading
import time
import uvicorn


def main():
    if not os.getenv("WORKBENCH_DESKTOP_TOKEN"):
        raise SystemExit("Desktop session token required")
    # Foundation Process already creates a process group on macOS.
    if os.getpgrp() != os.getpid():
        os.setsid()
    # Preserve system proxy routing for remote endpoints while keeping localhost local.
    from urllib.request import getproxies

    for scheme, proxy in getproxies().items():
        if scheme in ("http", "https", "all"):
            os.environ.setdefault(scheme.upper() + "_PROXY", proxy)
    # Keep domestic video traffic direct; model endpoints retain the system proxy.
    # A global proxy caused Bilibili 412 while OpenRouter required that proxy.
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1,.bilibili.com,.hdslb.com,.bilivideo.com,.bilivideo.cn"
    parent = os.getppid()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.set_inheritable(True)
    print(sock.getsockname()[1], flush=True)
    server = uvicorn.Server(
        uvicorn.Config("app.api.server:app", log_level="warning", access_log=False)
    )

    def watch_parent():
        while True:
            time.sleep(1)
            if os.getppid() != parent:
                os.killpg(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=watch_parent, daemon=True).start()
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()
