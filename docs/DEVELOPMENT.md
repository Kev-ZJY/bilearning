# 开发与运行

## 安装

需要 Python 3.11+、Node.js 22+、npm、ffmpeg / ffprobe。可选 OCR 使用 Tesseract 和相应语言包。macOS 原生外壳还需要 Xcode Command Line Tools。

macOS / Linux，从仓库根目录：

```sh
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend ci
```

Windows PowerShell，从仓库根目录：

```powershell
py -3 -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
npm --prefix frontend ci
```

ffmpeg / ffprobe 必须在服务的 PATH 中。Whisper 使用本地模型缓存，首次处理时可能需要下载权重；本仓库不附带模型、音频或示例课程。

## 浏览器开发模式

macOS / Linux 在 `backend/` 运行 `./run.sh`。脚本优先选择 `backend/.venv/bin/python`，也支持 `PYTHON` 指定解释器，读取可选的 `backend/.env`。

Windows PowerShell 的后端终端：

```powershell
cd backend
$env:STORAGE_ROOT = "../data"
.venv/Scripts/python.exe -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000
```

另一个终端从仓库根执行 `npm --prefix frontend run dev`，访问 `http://127.0.0.1:5180`。Vite 把 `/api` 转发到 `127.0.0.1:8000`。端口被占用时先停止占用服务，或同步修改 Vite 的代理和后端端口。

直接用 uvicorn 不会自动读取 `.env`；Windows 可在界面配置模型，或通过当前终端环境变量配置。脚本模式的默认存储目录为仓库根的 `data/`。

## 模型和存储配置

界面设置支持自己的 Chat Completions 兼容端点、模型 ID、密钥、输出预算和请求超时。并非每个兼容端点都完整支持工具调用、结构化输出或图像输入；具体任务需要的能力以调用代码为准。

| 变量 | 含义 |
|---|---|
| `LLM_BASE_URL` | 兼容接口根地址 |
| `LLM_MODEL` | 供应商的完整模型 ID |
| `LLM_API_KEY` | 使用者自己的密钥，保持在本机 |
| `LLM_MAX_TOKENS` | 输出预留，必须按模型实际限制设置 |
| `LLM_TIMEOUT` | 单次模型请求的超时秒数 |
| `STORAGE_ROOT` | 本机存储根目录 |
| `DB_URL` | 可选 SQLite 路径，未指定时跟随存储根目录 |
| `WHISPER_MODEL_DIR` | 可选本机模型目录 |
| `ASR_DEVICE` / `ASR_COMPUTE_TYPE` | 可选转写运行设置 |

浏览器模式中保存的模型设置可覆盖环境变量，密钥可保存在本机开发数据库。macOS 桌面模式使用 Keychain；桌面进程不读取开发 `.env`。详细数据流见 [隐私说明](PRIVACY.md)。

未知模型没有通用的上下文上限自动发现。若接入新模型，请核对 `agent/runtime/model_limits.py` 和实际供应商限制，避免照搬其他模型的预算。视频处理可以产生多次请求，费用由使用者的供应商账户承担。

## macOS 开发应用

完成依赖安装后，从仓库根执行：

```sh
python3 desktop/build.py --python backend/.venv/bin/python
open desktop/build/视频学习工作台.app
```

脚本构建前端和 Swift 外壳，应用引用当前源码和本机 Python。它是开发应用，不是包含运行时、签名和公证的独立安装包。不要分发生成的 `development.json`，其中包含构建机器的路径。

应用使用独立的本机应用索引；所选项目目录保存媒体、知识文件和 `.video-workbench/` 快照。关闭应用会停止其 Python 服务。不要让多个进程同时写同一项目。

## 构建前端

```sh
npm --prefix frontend run build
```

构建产物在 `frontend/dist/`，不提交到源码仓库。公开骨架不附带测试文件和测试专用依赖；这条命令运行 TypeScript 检查并打包前端。

## 可选 Docker

```sh
docker compose up --build
```

访问 `http://127.0.0.1:8080`。容器使用单 worker、独立数据卷和模型缓存，不继承宿主机模型密钥。目录选择、Keychain 等原生能力只属于 macOS 外壳。Docker 方式是可选运行配置，需要在使用者的目标环境验证；本仓库不包含预构建镜像。

## 排错

先查看任务首次失败的阶段和错误。区分依赖缺失、ffmpeg/模型权重不可用、网络与平台响应、模型权限/限额、结构化输出失败和文件写入失败。认证或额度错误应先修复配置，不要反复重试产生费用。

恢复入口会校验已有素材和候选身份。旧候选不兼容时需要重新生成知识候选；不要删除校验字段或伪造回执，也不应把下载、转写和抽帧当作每次重试的默认动作。
