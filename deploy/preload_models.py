"""在容器启动前手动预热 base / medium 权重，共用持久缓存卷。"""
from faster_whisper.utils import download_model

for name in ("base", "medium"):
    print(f"下载并缓存 Whisper {name}", flush=True)
    download_model(name)
print("模型缓存准备完成", flush=True)
