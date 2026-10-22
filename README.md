# bilearning

把 Bilibili 视频整理成可以阅读、检索和继续提问的知识文档。

这是一个面向个人学习与二次开发的本地应用骨架：React 工作台 + Python 后端 + 可选 macOS 原生外壳。你可以下载源码、配置自己的模型接口，研究从视频输入到知识生成、独立审核、文档保存和对话查询的完整调用链。

## 能做什么

- 按项目管理 Bilibili 完整视频链接和分 P，发起视频整理任务。
- 本地下载、Whisper 转写、逐秒画面准备与可选 OCR，再生成知识初稿。
- 对照来源检查覆盖情况，通过独立审查与作者修订后提交知识文档。
- 在同一工作区查看视频、知识目录、公开任务进度和助手对话。
- 检索知识点、用对话发起整理或改写，导出 HTML / Markdown / PDF。
- 在 macOS 开发应用中选择本地媒体和项目保存目录；浏览器模式提供跨平台开发入口。

这是可研究和扩展的源码骨架。视频质量、平台接口、模型能力和本机资源会影响结果；长视频可能需要较长处理时间和多次模型调用。生成内容需要结合原视频自行判断。

## 快速开始

准备 Python 3.11+、Node.js 22+、npm，以及可在命令行找到的 ffmpeg / ffprobe。Whisper 首次使用可能下载模型；OCR 可选安装 Tesseract 及需要的语言包。

```sh
git clone https://github.com/Kev-ZJY/bilearning.git
cd bilearning
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend ci
```

macOS / Linux 的终端一：

```sh
cd backend
./run.sh
```

终端二，从仓库根执行：

```sh
npm --prefix frontend run dev
```

打开 **http://127.0.0.1:5180**。后端使用 **127.0.0.1:8000**。在界面设置自己的 Chat Completions 兼容接口、模型 ID、密钥及输出预算；输出预算须符合所用模型的实际限制。未配置模型也可以启动界面。

需要环境变量配置时，将 `backend/.env.example` 复制为 `backend/.env` 并自行填写；不要提交真实 `.env`。Windows PowerShell 命令、macOS 应用构建和可选 Docker 运行见 [开发指南](docs/DEVELOPMENT.md)。

## 使用流程

1. 新建项目，将完整的 `bilibili.com/video/...` 链接粘贴到输入框，可带分 P 和整理要求。短链接请先展开。
2. 确认整理范围和选项后提交，在任务记录中查看实际阶段；发生错误时查看提示并修正配置。
3. 在知识区阅读正式文档。候选预览标为草稿；只有通过工作流提交的内容才是正式版本。
4. 围绕已有知识提问，或导出文档继续学习。已有素材能否复用由恢复校验判断。

## 项目结构

```text
backend/app/
  api/          HTTP、SSE 与桌面会话入口
  agent/        对话、模型运行时、知识生成、审核、媒体、工具、工作流
  services/     任务提交、文档保存、播放、项目文件和恢复
  worker/       下载、转写、抽帧、媒体检查、渲染
  core/         配置、SQLite、事件、取消、Keychain
  models/       数据模型
  sources/      Bilibili 平台接口
frontend/src/
  app/          工作区装配
  features/     chat / tasks / projects / knowledge / video / settings
  api/          共享 HTTP 客户端、类型和桌面桥
desktop/        Swift macOS 开发外壳
deploy/         可选容器模型准备工具
```

从 [架构与扩展入口](docs/ARCHITECTURE.md) 阅读调用链；配置和构建见 [开发指南](docs/DEVELOPMENT.md)，数据流见 [隐私说明](docs/PRIVACY.md)，依赖说明见 [第三方依赖](docs/THIRD-PARTY.md)。

## 数据和使用边界

项目以本地单用户使用为目标，默认服务只监听本机。模型调用会将相关转写、知识和必要证据发送到你配置的供应商；网页取证和 Bilibili 请求也需要网络。运行产生的数据库、媒体、文档和密钥不属于源码仓库。

请使用你有权访问和处理的视频，遵守平台规则与内容授权，尊重视频创作者。本项目与 Bilibili 无官方关联，不提供绕过访问限制或会员权限的机制。

公开仓库只包含应用源码、构建配置和使用文档，不包含测试文件、开发者数据、历史聊天、验收资料、媒体、模型权重或已配置的账号凭据。

## 许可

项目自身代码采用 [MIT License](LICENSE)，允许在保留版权与许可声明的前提下使用、修改和分发。第三方库、模型权重和视频内容分别遵循其自身许可与授权。
