# 本地语音应用

使用本目录的 React 客户端（`client/`），通过 WebRTC 与机器人对话：

- 语音识别：ElevenLabs Realtime STT（默认）或火山引擎双向流式识别
- 语言模型：DeepSeek
- 语音合成：ElevenLabs HTTP TTS（默认，模型为 `eleven_multilingual_v2`）或火山引擎双向流式合成

本目录保存应用的前端和后端：`bot.py` 是后端入口，`client/` 是独立的 Vite 网页客户端。
运行时需保留本目录所在的 Pipecat 仓库；它不是一个可单独搬走的前端构建产物。

## 启动

前后端是两个进程，都在项目根目录执行。后端：

```bash
bash local-voice-app/start.sh
```

客户端：

```bash
npm --prefix local-voice-app/client run dev
```

打开 <http://localhost:5174/>，连接并允许麦克风访问。两个进程各按 `Ctrl+C` 停止。

说到“启动应用”时，默认指后端加 `client/` 这个 React 客户端。Pipecat 自带的 Playground 仍由 runner 挂在
<http://localhost:7860/client/>，只在需要对照框架默认界面时使用，不是本应用的入口。
客户端连接的后端地址由 `VITE_BOT_START_URL` 决定，默认 `http://localhost:7860/start`，见 `client/env.example`；
后端换端口时浏览器仍会连旧地址。

脚本从文件位置定位项目根目录，读取根目录 `.env`，不会依赖临时目录。启动使用现有虚拟环境，不会自动同步或移除依赖。
客户端依赖装在 `client/node_modules`，不在版本控制中。

## 配置

根目录 `.env` 必须包含 DeepSeek 语言模型凭据；实际密钥不要提交到版本控制：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
```

默认识别和合成都使用 ElevenLabs，此时还需：

```dotenv
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_api_accessible_voice_id
```

音色必须允许当前账户通过 API 使用。语音供应商的 HTTP 会话支持系统代理环境变量。
改用火山引擎识别或合成后，该供应商不再需要 ElevenLabs 凭据。

默认使用 ElevenLabs 识别和合成。要改用火山引擎，在根目录 `.env` 中配置共享的 API 密钥和合成音色：

```dotenv
VOLCENGINE_API_KEY=your_volcengine_api_key
VOLCENGINE_TTS_SPEAKER=your_voice_id
```

识别和合成使用同一个 API 密钥。出现 `VOLCENGINE_API_KEY`（或 `VOLCENGINE_STT_OPTIONS`）时改用火山引擎识别；
出现 `VOLCENGINE_TTS_SPEAKER`（或 `VOLCENGINE_TTS_OPTIONS`）时改用火山引擎合成。只配置识别侧时，
合成继续使用 ElevenLabs，此时仍需 `ELEVENLABS_API_KEY` 和 `ELEVENLABS_VOICE_ID`。

模型版本由代码固定，不是配置项：识别用 `volc.seedasr.sauc.duration`（豆包流式语音识别模型 2.0
小时版），合成用 `seed-tts-2.0`（豆包语音合成大模型 2.0），两者都作为 `X-Api-Resource-Id`
请求头发送，必须是账户已开通的对应资源。

识别侧还可选配 `VOLCENGINE_STT_OPTIONS`
JSON 对象，支持 `enable_itn`（逆文本规范化）、`enable_punc`（标点）和
`corpus_context`（供应商定义的语料上下文对象）。省略标点或规范化开关时使用供应商默认行为。
`corpus_context` 序列化为 `request.corpus.context` 的 JSON 字符串，例如热词提示：

```dotenv
VOLCENGINE_STT_OPTIONS='{"corpus_context":{"hotwords":[{"word":"Pipecat"}]}}'
```

语料上下文内部格式和热词支持以[火山引擎接口文档](https://www.volcengine.com/docs/6561/1354869)
为准；应用校验 JSON 对象并传递上下文，不解释内部供应商字段。
第二遍识别固定开启：火山只在二遍模式下按静音分句并给出最终结果，关闭它就只能拿到中间结果，
用户的话永远不会被判定说完。`enable_nonstream` 等不作为配置项的选项会被拒绝。

合成音色必须通过 `VOLCENGINE_TTS_SPEAKER` 配置，不能出现在 `VOLCENGINE_TTS_OPTIONS` 中。
`VOLCENGINE_TTS_OPTIONS` 为可选 JSON 对象，支持 `audio_format`（音频容器，默认 `pcm`）、
`sample_rate`（输出采样率，省略时跟随流水线采样率）和 `additions`（供应商扩展，JSON 字符串）。
不支持的键会被拒绝。

选项和供应商在新会话装配时读取，不支持通话中切换。凭据、音色和识别选项只在后端使用，
浏览器沿用现有音频、字幕和指标通道。火山引擎配置不完整或选项不合法时，会话创建会返回明确错误，
错误不包含配置值。

`VOLCENGINE_RESOURCE_ID`、`VOLCENGINE_TTS_API_KEY` 和 `VOLCENGINE_TTS_RESOURCE_ID` 已不再使用，
配置其中之一会让会话创建失败并说明替代方式，避免留下一个看似生效的实际无用的设置。
音色不要填进 `VOLCENGINE_RESOURCE_ID`，那是已移除的识别模型版本设置。

示例配置见根目录 `env.example` 和 `local-voice-app/bot/.env.example`。
`local-voice-app/bot/bot.py` 是共享入口的兼容转发，也读取根目录 `.env`。


## 知识库

应用经 HTTP 接入外部知识库：把知识库查询注册成模型可自主调用的一个工具，由模型判断该不该查。
整项功能由根目录 `.env` 中的一个变量启用：

```dotenv
WIKI_TOKEN=your_wiki_api_token
```

不配置这个变量，或者把它留空，会话就不接知识库：不注册工具，指令也不提知识库，行为与没有这项功能时完全一致。

知识库是 `llm_wiki` 桌面应用，它在本机回环上暴露一个 token 保护的 HTTP API，`WIKI_TOKEN` 是该 API 的令牌。
API 根地址由可选变量 `WIKI_API_BASE_URL` 给出，默认 `http://127.0.0.1:19828`，必须是绝对的 http 或 https 地址。
应用只请求它的两个路由：`POST /api/v1/projects/current/search`（带 `includeContent: true`，
一次取回排名靠前的若干页及其正文）和 `GET /api/v1/projects/current/files/content`
（搜索没带回正文时按 path 补读一页）。

对模型只暴露一个工具，它内部完成「搜索 → 读最相关的一两页 → 返回正文」，这是一次调用。
搜索只用来定位页面，回答取自整页正文——片段会在行中间截断，答案所在的那一行往往不在其中。
读几页、每页留多长、一次检索的上限，都是应用内的常量而不是配置项。

知识库没启动、令牌不被接受、检索超时或没查到内容，对用户一律表现为“查不到”：模型被要求照实说没找到，
不报告故障，也不拿自己已知的内容顶上，会话继续。日志把“没查到”和“没连上”分开记。
会话建立时不连知识库，它当时是否可用要到第一次检索才知道。`WIKI_API_BASE_URL` 不是绝对地址则让会话创建失败，
错误说明原因且不回显配置值。

本机使用时，bot、浏览器客户端和知识库桌面应用**三者必须同机**：HTTP API 只监听回环。远程部署要另找出路。

知识库访问不走代理：它的 HTTP 会话显式 `trust_env=False`，本机环境里存在代理时，回环请求也不会被交给代理。
语音供应商的会话不受此影响。

检索进行期间先播一句“我查一下。”。它是应用内的常量，不是配置项；不进入对话历史，
但会作为机器人转写显示在客户端。模型只在知识库可用时才去查，寒暄、感谢、闲聊不触发检索。

`local-voice-app/scenarios/knowledge_lookup.yaml` 覆盖这条路径，文件头注释里有完整运行命令与前置条件。


## 新环境准备

当前机器已经安装依赖和 NLTK 数据，无需重复操作。在新的仓库检出中执行：

```bash
uv sync --extra runner --extra webrtc
uv run --no-sync python -m nltk.downloader punkt_tab
npm --prefix local-voice-app/client install
```

知识库功能只用 HTTP 客户端，不需要可选依赖。

若使用离线下载的 `punkt_tab.zip`，将其解压到 `~/nltk_data/tokenizers/`，确保存在 `~/nltk_data/tokenizers/punkt_tab/english/`。

框架的 Python 包不包含本应用目录。部署时需一并提供本目录、框架依赖及运行时环境变量。

## 双供应商识别评估

无需真实凭据的完整流水线验证：

```bash
uv run --no-sync pytest tests/test_local_voice_eval.py tests/test_local_voice_app.py tests/test_volcengine_stt.py tests/test_elevenlabs_stt.py
```

测试通过真实评估传输重放同一段录音，保留应用的语音检测、轮次管理、字幕观察器和供应商服务，
将外部识别、语言模型、语音合成接口替换为本地测试端点。验证中间字幕、确定分句、
静音后恢复、机器人说话时打断、相同短句再次提交以及客户端取消后的连接清理。
火山引擎端点会重发同一确定分句，确认它只提交一次；新一轮相同文字仍正常提交。
第二遍识别开启通过发往火山引擎端点的真实请求检查，服务测试另行覆盖正常结束时的尾包。

配置可用时，在仓库根目录运行真实基线：

```bash
uv run --no-sync python local-voice-app/eval_stt.py
```

脚本顺序启动两个供应商的完整应用，使用独立端口 `17860`，每次结束后回收进程。
它读取根目录 `.env`，显式进程环境变量优先，不修改配置文件。ElevenLabs 评测子进程会移除全部火山引擎配置，
因为共享的 API 密钥同时决定两侧供应商；火山引擎评测子进程保留该配置，并去掉已移除的设置。
两家都需要 DeepSeek 凭据；ElevenLabs 分支需要可用的 ElevenLabs 凭据及音色，火山引擎分支需要自己的 API 密钥和音色。
缺少配置的供应商写为 `skipped`，不计作通过，也不填写准确率或延迟。
已有配置但鉴权失败、等待不到机器人发声、缺少字幕或打断事件等情况写为 `failed`，命令返回非零。

只运行一家、调整超时或结果路径：

```bash
uv run --no-sync python local-voice-app/eval_stt.py --provider volcengine --timeout 30 --output .local/stt-eval/volcengine.json
```

默认结果位于 `.local/stt-eval/report.json`，该目录已忽略，不提交录音转写等运行数据。
报告包含录音路径及 SHA-256、参考文字、每轮原始字幕/说话事件、行为断言、字符准确率和最终延迟。
默认共享录音为 `scripts/release-evals/assets/capital_question.wav`，参考文字为
“What is the capital of Germany?”。可用 `--audio /path/to/mono.wav --reference "reference words"`
替换为单声道 16 位 PCM WAV 和对应参考文字；两家必须使用同一文件及参考。

每次运行先等待欢迎语结束，连续发送两秒静音，再播放录音；机器人开始回复后立即再次播放同一录音，
最后继续发送静音并观察字幕。原始 RTVI 消息保留 `final=false` 的中间字幕；普通场景文件的
`user_transcription` 事件只保留最终文字，不能单独证明中间字幕已送达。

字符错误率使用 NFKC、大小写归一、去除非字母数字字符后的编辑距离除以参考字符数；
字符准确率为 `max(0, 1 - 字符错误率)`。每轮所有确定分句按到达顺序拼接，并单列完整文字是否匹配参考。
一轮允许多个确定分句；报告不会按文字去重，也不会把分句数量直接解释为重复提交。
RTVI 不携带供应商分句身份，真实运行只能核对完整参考文字；精确的重发抑制由本地协议测试验证。

最终延迟取同轮**最后一条最终字幕到达时间减去最后一条原始 VAD 停止事件到达时间**，单位毫秒，
不包含整段录音时长。它是客户端观察到的 VAD 停止后延迟，包含传输及处理开销，
没有扣除 VAD 自身的停止判定窗口，不等于人工标注语音结束点的延迟。
缺少任一事件或出现负时差时写为 `null`、`unavailable`，不补零。
收尾使用 `eval-cancel` 并检查连接及进程结束；正常 `EndFrame` 的供应商尾部刷新由服务测试单独验证。

末尾观测窗口固定为两秒；`--timeout` 控制所需事件的最长等待，不会延长该窗口。
晚于观测窗口才到达的分句可能导致结果不完整或参考文字匹配失败，应结合原始事件分析，
不能把一次通过解释为覆盖任意晚到结果。默认短录音的两轮结果用于小样本回归基线，不能代表通用识别准确率。
