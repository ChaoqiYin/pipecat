# 本地语音应用

使用 Pipecat 自带的 Playground 网页，通过 WebRTC 与机器人对话：

- 语音识别：ElevenLabs Realtime STT
- 语言模型：DeepSeek
- 语音合成：ElevenLabs HTTP TTS，模型为 `eleven_multilingual_v2`

本目录保存应用启动入口。网页由 `pipecat-ai-prebuilt` 依赖提供，无需单独启动前端构建服务。运行时需保留本目录所在的 Pipecat 仓库；它不是一个可单独搬走的前端构建产物。

## 启动

在项目根目录执行：

```bash
bash local-voice-app/start.sh
```

打开 <http://localhost:7860/client/>，连接并允许麦克风访问。终端按 `Ctrl+C` 停止服务。

脚本从文件位置定位项目根目录，读取根目录 `.env`，不会依赖临时目录。启动使用现有虚拟环境，不会自动同步或移除依赖。

## 配置

根目录 `.env` 必须包含以下变量；实际密钥不要提交到版本控制：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_api_accessible_voice_id
```

音色必须允许当前账户通过 API 使用。HTTP 会话支持系统代理环境变量。

## 新环境准备

当前机器已经安装依赖和 NLTK 数据，无需重复操作。在新的仓库检出中执行：

```bash
uv sync --extra runner --extra webrtc
uv run --no-sync python -m nltk.downloader punkt_tab
```

若使用离线下载的 `punkt_tab.zip`，将其解压到 `~/nltk_data/tokenizers/`，确保存在 `~/nltk_data/tokenizers/punkt_tab/english/`。

框架的 Python 包不包含本应用目录。部署时需一并提供本目录、框架依赖及运行时环境变量。
