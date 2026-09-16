# 本地语音应用

使用 Pipecat 自带的 Playground 网页，通过 WebRTC 与机器人对话：

- 语音识别：ElevenLabs Realtime STT（默认）或火山引擎双向流式识别
- 语言模型：DeepSeek
- 语音合成：ElevenLabs HTTP TTS（默认，模型为 `eleven_multilingual_v2`）或火山引擎双向流式合成

本目录保存应用启动入口。网页由 `pipecat-ai-prebuilt` 依赖提供，无需单独启动前端构建服务。运行时需保留本目录所在的 Pipecat 仓库；它不是一个可单独搬走的前端构建产物。

## 启动

在项目根目录执行：

```bash
bash local-voice-app/start.sh
```

打开 <http://localhost:7860/client/>，连接并允许麦克风访问。终端按 `Ctrl+C` 停止服务。

脚本从文件位置定位项目根目录，读取根目录 `.env`，不会依赖临时目录。启动使用现有虚拟环境，不会自动同步或移除依赖。

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

音色必须允许当前账户通过 API 使用。HTTP 会话支持系统代理环境变量。
改用火山引擎识别或合成后，该供应商不再需要 ElevenLabs 凭据。

默认使用 ElevenLabs 识别。要改用火山引擎，在根目录 `.env` 中配置其 API 密钥和账户已开通的资源 ID：

```dotenv
VOLCENGINE_API_KEY=your_volcengine_api_key
VOLCENGINE_RESOURCE_ID=your_volcengine_resource_id
VOLCENGINE_STT_OPTIONS='{"enable_itn":true,"enable_punc":true}'
```

未配置任何 `VOLCENGINE_*` 识别设置时，应用继续使用 ElevenLabs 识别。出现任一火山引擎设置后，API 密钥和资源 ID 都必须提供；资源 ID 不提供代码默认值，必须是账户已开通的识别资源。`VOLCENGINE_STT_OPTIONS`
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
浏览器沿用现有音频、字幕和指标通道。火山引擎配置不完整或选项不合法时，会话创建会返回明确错误，
错误不包含配置值。

默认使用 ElevenLabs HTTP 合成。要改用火山引擎合成，在根目录 `.env` 中配置其 API 密钥、账户已开通的资源 ID 和音色 ID：

```dotenv
VOLCENGINE_TTS_API_KEY=your_volcengine_api_key
VOLCENGINE_TTS_RESOURCE_ID=seed-tts-2.0
VOLCENGINE_TTS_SPEAKER=your_voice_id
VOLCENGINE_TTS_OPTIONS='{"audio_format":"pcm","sample_rate":24000}'
```

未配置任何 `VOLCENGINE_TTS_*` 设置时，应用继续使用 ElevenLabs 合成，并需要 `ELEVENLABS_API_KEY` 和
`ELEVENLABS_VOICE_ID`。出现任一火山引擎合成设置后，API 密钥、资源 ID 和音色都必须提供；
资源 ID 不提供代码默认值，必须是账户已开通的合成资源。音色单独通过 `VOLCENGINE_TTS_SPEAKER`
配置，不能出现在 `VOLCENGINE_TTS_OPTIONS` 中。
`VOLCENGINE_TTS_OPTIONS` 为可选 JSON 对象，支持 `audio_format`（音频容器，默认 `pcm`）、
`sample_rate`（输出采样率，省略时跟随流水线采样率）和 `additions`（供应商扩展，JSON 字符串）。
不支持的键会被拒绝。

语音合成与语音识别的供应商选择互不影响：`VOLCENGINE_TTS_*` 不改变识别供应商，
`VOLCENGINE_API_KEY` / `VOLCENGINE_RESOURCE_ID` 也不改变合成供应商。两家可以任意组合，
包括全部使用火山引擎时不提供任何 ElevenLabs 凭据。

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

## 双供应商识别评估

无需真实凭据的完整流水线验证：

```bash
uv run --no-sync pytest tests/test_local_voice_eval.py tests/test_local_voice_app.py tests/test_volcengine_stt.py tests/test_elevenlabs_stt.py
```

测试通过真实评估传输重放同一段录音，保留应用的语音检测、轮次管理、字幕观察器和供应商服务，
将外部识别、语言模型、语音合成接口替换为本地测试端点。验证中间字幕、确定分句、
静音后恢复、机器人说话时打断、相同短句再次提交以及客户端取消后的连接清理。
火山引擎端点会重发同一确定分句，确认它只提交一次；新一轮相同文字仍正常提交。
第二遍识别关闭通过发往火山引擎端点的真实请求检查，服务测试另行覆盖正常结束时的尾包。

配置可用时，在仓库根目录运行真实基线：

```bash
uv run --no-sync python local-voice-app/eval_stt.py
```

脚本顺序启动两个供应商的完整应用，使用独立端口 `17860`，每次结束后回收进程。
它读取根目录 `.env`，显式进程环境变量优先，不修改配置文件。ElevenLabs 评测子进程会移除火山引擎识别配置；火山引擎评测子进程保留该配置。
完整应用仍需要 DeepSeek、ElevenLabs 合成凭据及可访问的音色；火山引擎还需要自己的识别凭据。
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
