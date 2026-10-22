# 第三方依赖

仓库只分发项目自身源码及配置，不附带 node_modules、Python 环境、媒体工具、模型权重或视频内容。项目自身代码采用根目录中的 MIT License；第三方材料继续适用其自身许可。

前端主要依赖 React、React DOM、marked、Vite、TypeScript 和 React 插件。完整版本和依赖树见 `frontend/package-lock.json`。

后端主要依赖 FastAPI、Uvicorn、SQLAlchemy、faster-whisper / CTranslate2、requests、HTTPX、Pydantic、ReportLab 和 Pillow。声明见 `backend/requirements.txt`；Python 依赖部分使用版本范围，安装时请检查实际解析版本及随附 LICENSE。

macOS 外壳使用系统 AppKit、WebKit 和 Security.framework。ffmpeg / ffprobe、可选 Tesseract 及 Whisper 权重由使用者自行安装或下载，它们不因项目采用 MIT 而改变许可。尤其在制作包含运行时或媒体工具的二进制安装包时，应另行核对实际构建版本与分发义务。

视频、字幕、截图、网页和生成文档中的来源内容不归项目代码许可证授权。请取得相应的访问和使用权限，并尊重原作者署名与授权条件。
