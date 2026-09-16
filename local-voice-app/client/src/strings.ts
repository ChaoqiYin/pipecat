/**
 * 界面中文文案的唯一来源。
 *
 * 只收录我们自己渲染的文字。下面这些英文写死在 `@pipecat-ai/voice-ui-kit` 的打包
 * 产物里，没有任何属性出口，因此改不了，界面上的调试面板也刻意保持英文：
 *
 * - Events 调试面板整体（含标题与筛选框）
 * - 对话里函数调用折叠块的 `Arguments` / `Result`
 * - 事件日志正文与事件类型名
 */

export const STRINGS = {
  app: {
    debugToggle: '切换调试面板',
    errorCount: '错误数量：',
  },
  tabs: {
    conversation: '对话',
    metrics: '指标',
  },
  textRenderMode: {
    karaoke: '逐字高亮',
    captions: '字幕',
    instant: '即时',
  },
  textInput: {
    placeholder: '输入消息…',
    noConnected: '连接后可发送',
  },
  conversation: {
    connecting: '正在连接机器人…',
    notConnected: '未连接到机器人',
    notConnectedHint: '连接后即可实时查看对话内容。',
    unsupported: '服务器不支持 BotOutput 事件',
    unsupportedHint: '需要 RTVI 1.1.0 及以上版本，否则无法显示对话内容。',
    waiting: '等待消息…',
    roles: {
      assistant: '助手',
      client: '用户',
      system: '系统',
      functionCall: '函数调用',
    },
  },
  botAudio: {
    title: '机器人音频',
    noAudio: '无音频',
    volumeLabel: '机器人音量',
  },
  connection: {
    client: '客户端',
    agent: '机器人',
  },
  /** 对应 `@pipecat-ai/client-js` 的 `TransportStateEnum`。 */
  transportState: {
    disconnected: '未连接',
    initializing: '初始化中',
    initialized: '已初始化',
    authenticating: '认证中',
    authenticated: '已认证',
    connecting: '连接中',
    connected: '已连接',
    ready: '就绪',
    disconnecting: '断开中',
    error: '错误',
  },
  connectButton: {
    connect: '连接',
    initializing: '初始化中…',
    connecting: '连接中…',
    disconnect: '断开',
    disconnecting: '断开中…',
    error: '错误',
  },
  errorCard: {
    title: '出错了',
  },
  metrics: {
    waiting: '等待指标数据…',
    tokenUsage: '令牌用量',
    promptTokens: '输入令牌',
    completionTokens: '输出令牌',
    totalTokens: '令牌总数',
    ttfb: '首字节延迟',
    processing: '处理耗时',
    characters: '字符数',
  },
} as const;
