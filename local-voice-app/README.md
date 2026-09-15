# 本地语音应用

使用 Pipecat 自带的 Playground 网页，通过 WebRTC 与机器人对话：

- 语音识别：ElevenLabs Realtime STT（默认）或火山引擎双向流式识别
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

`STT_PROVIDER` 只接受 `elevenlabs` 和 `volcengine`，未设置时使用 `elevenlabs`。
切换到火山引擎时，在根目录 `.env` 中增加：

```dotenv
STT_PROVIDER=volcengine
VOLCENGINE_API_KEY=your_volcengine_api_key
VOLCENGINE_RESOURCE_ID=volc.seedasr.sauc.duration
VOLCENGINE_STT_OPTIONS='{"enable_itn":true,"enable_punc":true}'
```

资源 ID 必须是账户已开通的识别资源；省略时使用上述默认值。`VOLCENGINE_STT_OPTIONS`
为可选 JSON 对象，支持 `enable_itn`（逆文本规范化）、`enable_punc`（标点）和
`corpus_context`（供应商定义的语料上下文对象）。省略标点或规范化开关时使用供应商默认行为。
`corpus_context` 序列化为 `request.corpus.context` 的 JSON 字符串，例如热词提示：

```dotenv
VOLCENGINE_STT_OPTIONS='{"corpus_context":{"hotwords":[{"word":"Pipecat"}]}}'
```

语料上下文内部格式和热词支持以所选资源的[火山引擎接口文档](https://www.volcengine.com/docs/6561/1354869)
为准；应用校验 JSON 对象并传递上下文，不解释内部供应商字段。
第二遍识别固定关闭，`enable_nonstream` 等未支持的选项会被拒绝。

选项和供应商在新会话装配时读取，不支持通话中切换。凭据、资源 ID 和识别选项只在后端使用，
浏览器沿用现有音频、字幕和指标通道。所选供应商缺少凭据、供应商名称无效或选项不合法时，
会话创建会返回明确错误，错误不包含配置值。
即使使用火山引擎识别，语音合成仍需 `ELEVENLABS_API_KEY` 和 `ELEVENLABS_VOICE_ID`。

示例配置见根目录 `env.example` 和 `local-voice-app/bot/.env.example`。
`local-voice-app/bot/bot.py` 是共享入口的兼容转发，也读取根目录 `.env`。


## 新环境准备

当前机器已经安装依赖和 NLTK 数据，无需重复操作。在新的仓库检出中执行：

```bash
uv sync --extra runner --extra webrtc
uv run --no-sync python -m nltk.downloader punkt_tab
```

若使用离线下载的 `punkt_tab.zip`，将其解压到 `~/nltk_data/tokenizers/`，确保存在 `~/nltk_data/tokenizers/punkt_tab/english/`。

框架的 Python 包不包含本应用目录。部署时需一并提供本目录、框架依赖及运行时环境变量。
