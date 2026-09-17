# 知识库查询交给模型自主调用，而不是用 pipecat.flows

Pipecat 自带 `pipecat.flows`（前身是独立的 `pipecat-ai-flows` 包），用 YAML 或 JSON 定义对话节点和跳转。应用不用它：知识库查询注册成普通 LLM 工具，由模型自己判断该不该查。

## 考虑过的其他做法

- **单节点 Flows + `global_functions`**：Flows 的确定性只作用在「跳到哪个节点」，不作用在「要不要调这个工具」，所以模型自主判断能保住；`@flows_tool_options` 也是官方的超时途径。但它的 handler 是另一套契约——Flows 按参数个数分发、把参数以 dict 传入，而工具 handler 接受 `FunctionCallParams`——接入要为此写一层适配。Flows 提供的两样东西是节点跳转和流程状态，本应用都不需要。
- **每轮用户说完强制预查一次**：放弃模型的自主权，代价是给每个回合都叠加一次检索往返，包括寒暄。

## 后果

- 不新增处理器：工具挂在 `LLMContext` 上，现有钉死处理器顺序的测试不受影响。
- 工具超时在 handler 的注册上给出（`register_function(..., timeout_secs=...)`）：`FunctionSchema` 上没有 timeout 字段，随 schema 一起被自动注册的 handler 只能沿用服务级默认值，所以工具只以 schema 暴露给模型，handler 另行注册。
- 打断时取消在途检索是框架默认行为，不需要额外代码。
- 将来若要做有阶段的对话流程，`pipecat.flows` 是框架给的答案。
