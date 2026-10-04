# Tradar

Tradar 是一个正在建设中的 Agentic Quantitative Research Platform。目标不是让 Agent 写更多代码，而是让量化研究能够被稳定地理解、检索上下文、调用确定性研究能力、验证、解释、记录并复用，形成可复现、可评估、可追溯的人机协同研究系统。

## 项目目标 / Target

```text
User Question
  → Safety
  → Model (tool_choice=auto) / On-demand Knowledge or History Lookup
  → Selected Research Action / Deterministic Read-only Proceed
  → Deterministic Execution / ResearchRun
  → Bounded ToolResult Evidence
  → Answer Synthesis
  → Cited-Evidence Grounding
  → Final Answer + Telemetry
```

`research/` 是 deterministic quantitative engine；`agent/` 是负责编排、能力调用和评估的 harness/orchestration 层。


## 项目边界

Tradar 聚焦于量化研究流程的 Agent 化，不试图覆盖投资决策、交易执行或底层数据基础设施。

明确边界：

- 不推荐股票：不输出面向用户的个股买卖建议、目标价或仓位建议。
- 不推荐策略：可以研究、比较和评估策略，但不向用户给出“应当采用某策略进行投资”的投资建议。研究结论以可复现的实验结果、风险指标和适用条件为主。
- 不负责交易执行：不连接券商下单，不管理真实账户、资金或仓位，也不执行 autonomous trading。
- 不负责市场数据管线：行情、财务和其他研究数据由外部数据基础设施准备，Tradar 只消费规范化后的数据输入。
- 不把 LLM 当作量化计算引擎：因子计算、回测、指标和研究结果由 research/ 中的确定性代码完成；Agent 负责理解、检索、编排、调用、评估和解释。
- 不把探索性代码默认产品化：一次性研究可以存在于 scripts/；只有稳定、可复用、语义明确的能力才沉淀为 research/ 能力或 Agent Tool。

核心职责仅限：

`Research Question → Context / Retrieval → Tool Orchestration → Deterministic Research → Evaluation / Interpretation → Research Memory / Human Review`


## 当前能力 / Current Capabilities

| 能力 | 状态 |
| --- | --- |
| Tool Calling 与 fixture eval | 已落地 |
| ResearchRun State Lifecycle 与 Deterministic Multi-step Executor | 已落地 |
| Context Selection / Compaction / Construction 与 eval | 已落地 |
| Research Record corpus | 已落地 |
| Lexical / Semantic Retrieval（含 CJK character bigram）与 Retrieval Eval（baseline / semantic / abstention） | 已落地 |
| Relevance Verification Eval 与 Verified Semantic Retrieval runtime | 已落地 |
| Model-first native tool loop；旧 Planning contract/eval 独立保留 | 已落地 |
| Grounding contract/eval 与 runtime verifier | 已落地 |
| Memory write/update/ignore decision eval、append-only lifecycle store、recall eval 与 runtime recall | 已落地 |
| HITL approval gate eval、runtime gate 与 minimal approval lifecycle | 已落地 |
| AE-09 whole-system Agent runtime/eval 与 failure attribution | 已落地 |
| AE-10 Answer synthesis 与实际 ToolResult evidence output binding | 已落地 |
| AE-11 provider-call telemetry 与 run-level observability | 已落地 |
| AE-12 runtime safety boundary 与 indirect-injection quarantine | 已落地 |
| AE-14 Qdrant hybrid knowledge retrieval（BM25 + dense / RRF / canonical hydration / verification）与 opt-in runtime | 已落地 |
| Related research context 与 verified answer evidence 分离（model-first runtime） | 已落地 |
| `research/` quantitative engine、factor analysis 与 backtest | 已落地 |

`agent/agent.py::run_agent` 是 safety → model（`tool_choice=auto`）→ tools → model 的有界循环。普通问题直接回答；`request_clarification` 立即澄清；研究 Tool 被选中后才进入确定性执行与 `ResearchRun`。当前四个 local/read-only research tools 确定性 proceed，不另调用 HITL 模型；未来 external/destructive/production/financial actions 保留基于具体 tool/arguments 的审批 gate。模型看实际有界 tool observations 决定下一步；完成研究后只用真实 ToolResult evidence 做 synthesis/cited-evidence grounding。没有独立 preflight/planner/router，也没有另一套“模型失败后转 Agent”系统。Memory store、context selector/compactor、旧 planner/loop evaluator 仍是独立能力，不在主 runtime 热路径；approval resume 未实现。

普通 chat/meta 是一次 model call、零研究阶段，不做研究 grounding；provider/格式错误显式返回，不再用另一个路由阶段掩盖。初始模型上下文只放最近最多 4 条、合计 2,000 个 history content 字符；`lookup_history` 一次性读取初始 slice 未包含的 older-only 安全 history，同样最多 4 条/2,000 字符，超出会标记 truncation。研究 synthesis 仍可使用完整安全 history，但 history 不会成为 evidence。Telemetry 保留 static provider discovery，避免 Mock 动态 `.client` 链导致 OOM。

Telemetry 记录 provider/model/stage、provider 返回的 usage tokens、provider latency 与完整 runtime wall-clock，并使用版本化配置估算成本，同时保留 per-stage、failure/terminal attribution。Safety 将 system/product rules 视为可信指令，将 user/retrieval/tool 内容视为不可信数据；明显不安全请求和 active indirect injection 会被确定性 block/quarantine，destructive action 仍经过 HITL。

DeepSeek / OpenAI 远程 SDK 调用关闭 SDK 隐式重试：每次尝试最多 20 秒（含完整响应读取），总调用预算 45 秒，瞬态连接/超时、408/409/429/5xx 最多重试一次。无 `Retry-After` 时等待 250–500 ms；有效的服务端等待超过 1 秒时直接返回错误，不提前重试。空响应/无效 provider envelope 不重试；模型内容不符合阶段 schema 仍由原有 parser/repair contract 处理。错误区分 `provider_timeout`、`provider_rate_limit`、`provider_empty_response`、`provider_invalid_response`，其它操作失败保留 `provider_error`，不会变为语义 abstain。Telemetry 增加 attempt/retry 计数及各 attempt latency/backoff；call latency 与 per-stage `provider_latency_ms` 包含重试等待。失败尝试没有 usage 时，整次调用的累计 tokens/cost 保持 unknown，不当作零；Ollama 本地 embedding 策略不变。

CLI 知识检索默认关闭；`--retrieval qdrant` 仅暴露按需 `search_knowledge`，不会在模型前访问 Qdrant/Ollama。HTTP 默认暴露同一 dense capability，history/锁/响应 shape 不变；`--retrieval-strategy hybrid` 保留 BM25/RRF，legacy `ollama`/`openai` 也按需运行。模型调用 lookup 后得到 quarantine 过的 compact related context；如果继续做新研究，跳过 verifier 且 context 不进入 evidence。若只基于历史记录结束，则 batched relevance verification → supported full-record evidence → synthesis/grounding；不支持则 abstain，基础设施/verifier 错误显式 fail closed。Runtime 不自动建索引。见 [retrieval README](resources/retrieval/README.md#agent-runtime-opt-in)。

Deterministic safety gate 同时覆盖常见英文攻击及中文直接指令：忽略规则、暴露系统提示词、读取/输出密钥或凭据、绕过安全/审批。中文引号中的分析材料及非执行性的安全讨论不视为指令；这是有界命令模式，不是通用多语言分类器。

模型选择知识 lookup 后，连续 lookup 的安全记录按 `research_id` 累积去重（history lookup 不清空记录）；全局保留不超过 candidate_limit（最多 5 条），最新 brief snapshot 不超过 6,000 字符，旧 snapshots 只保留 superseded 标记。超预算时淘汰最早保留的记录，不按跨 query score 重排。每个 full verifier record 最多 12,000 字符，不截断研究事实；严格 relevance verification 仅使用一个 answer target，默认原始 turn。模糊追问由模型依据有界安全 history 显式设置 `search_knowledge.answer_target`（最多 2,000 字符），保留原始问题的 subject/date/scope；检索 queries 不自动成为答案约束。若模型结束历史答案路径，只有 verifier 明确接受的记录才可转换为有界 `knowledge_record` evidence，保留 `knowledge-<research_id>` 引用和文件 hash/date/status 等 provenance；不创建 ResearchRun 或执行研究。无支持则 controlled abstain，不从 related context 拼凑答案。相关记录不代表完整支持：synthesis 可 abstain/error，grounding 不支持的答案会被 block，loop trace 如实记录最终 outcome。`run_agent(client=...)` 是主接口；旧 `planner_client` 参数及 null `plan`/`observed.planning`/`observed.orchestration` 字段仅作已弃用兼容，不代表运行阶段。

Authored Research Experiments 在隔离的只读数据/code sandbox 内执行：4 GiB 地址空间、80 秒 CPU / 90 秒 wall-time，单进程、64 个文件描述符及 64 KiB 输出限制。Worker 独立配置 DuckDB 为 1 thread / 256 MiB memory，spill 位于 `/work/cache/duckdb`（最多 256 MiB）；整个 `/work` 为 512 MiB 有界 tmpfs。共享/生产 DuckDB 默认值不变。仅 worker 通过 `TRADAR_EXPERIMENT_SANDBOX=1` 启用 bounded cold-window 复权读取：保留全历史事件累计口径，但不构建全历史 daily-factor cache；前复权按每个 symbol 窗口内最后交易日取单个 anchor，避免对宽价格结果执行全表 window。sandbox 外仍保留共享 cache rebuild / force-refresh 语义。Authoring 必须只加载需要的字段/掩码并及时释放宽表中间结果，逐个 horizon 计算/汇总/释放，不能缩短研究窗口或吞掉资源错误。超限保留结构化执行错误；这些限制不保证任意规模研究都能完成。

可检查的本地 runtime artifacts 统一放在 gitignored `.runtime/`：生成的实验源码保留于 `.runtime/experiments/<source_sha256>.py`（`TRADAR_EXPERIMENT_ARTIFACT_DIR` 显式覆盖仍有效），知识索引状态保留于 `.runtime/knowledge_index/`。`python -m agent.tools.eval --provider fixture --repeats 1` 生成的 weights/close/open CSV 位于 `.runtime/agent_eval/`，可作为人工 UAT 的 runtime artifact 参数；不要引用临时 `/tmp` fixture。需要保存 CLI/UAT 输出时，使用 `.runtime/runs/` 下的报告文件（CLI 仍输出到 stdout，不新增自动持久化）。实验 worker 的临时 isolation 工作目录和测试临时文件仍用完清理；canonical market data 的 `DATA_PATH` 与已有 provenance/显式 artifact 路径不变。

## 架构快照 / Architecture Snapshot

```text
User Question
     │
     ▼
agent/                         resources/                 tests/
├── agent.py                   ├── eval/                  └── regression tests
├── agent_eval.py              ├── prompts/
├── core/                      └── knowledge/
├── answer/
├── retrieval/
├── context/
├── planning/
├── grounding/
├── memory/
├── hitl/
└── tools/
     │  harness / orchestration
     ▼
research/  ───────────────────► utils/
deterministic quantitative       data loading / infrastructure
engine
```

`resources/` 只存放非执行型 dataset、prompt 和 Research Record corpus；`tests/` 保存公共能力的回归测试。依赖方向保持为 `agent → research → utils`。

## 项目结构

```text
tradar/
├── agent/
│   ├── agent.py            # 集成 runtime：request → final answer + telemetry
│   ├── agent_eval.py       # AE-09 whole-system deterministic eval
│   ├── core/               # Contracts、providers、resources、telemetry、safety
│   ├── answer/             # Answer synthesis 与 eval
│   ├── tools/              # Tool schemas、calling、research tools、executor 与 Tool Calling eval
│   ├── context/            # Selector、compactor、builder 与 Context Eval runner
│   ├── retrieval/          # Research Record loader、lexical/semantic retrieval、verification 与 eval
│   ├── planning/           # Standalone planner/orchestrator contract 与 eval，非主 runtime
│   ├── grounding/          # Claim ↔ Evidence verifier 与 Grounding eval
│   ├── memory/             # Decision、atomic store、recall 与对应 eval
│   └── hitl/               # Approval gate、approval lifecycle 与 HITL eval
├── research/               # 唯一的量化研究引擎
│   ├── panel.py            # 面板数据与 universe
│   ├── factor_analyzer.py  # 因子分析
│   ├── factor_dsl.py       # 因子 DSL
│   ├── metrics.py          # 绩效指标
│   ├── vector_backtest.py  # 向量化回测
│   └── ...
├── scripts/                # 研究实验、回测和一次性任务
├── utils/                  # 数据加载与通用基础设施
├── data/                   # Tradar 自身的过程数据和研究缓存
├── resources/
│   ├── eval/               # JSON datasets；不放执行代码
│   ├── prompts/            # 稳定的 eval/system prompt
│   └── knowledge/          # Research Records 与 retrieval corpus，不含私有历史
└── tests/                  # 公共能力回归测试
```

## Roadmap

- Thin runnable entrypoint + a few real-provider smoke runs
- 根据 smoke 结果决定是否接入 compactor、推导 truthful HITL action、增加 approval resume/replan，或整合 memory

## 快速开始

注意环境变量 `DATA_PATH` 是 `tradar.duckdb` 存储的日线等数据目录，Tradar 本身只保留研究过程数据和缓存。相关的数据结构（表和关系）见 `utils/duckdb_manager.py`，使用前可以做相应的迁移/适配。

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

本地 deterministic eval（不会调用远程模型）：

```bash
python -m agent.tools.eval --provider fixture --repeats 1
python -m agent.context.eval --provider fixture --repeats 1
python -m agent.retrieval.eval
python -m agent.retrieval.relevance_verifier_eval --provider fixture --repeats 1
python -m agent.answer.eval --provider fixture --repeats 1
python -m agent.planning.eval --provider fixture --repeats 1
python -m agent.grounding.eval --provider fixture --repeats 1
python -m agent.memory.eval --provider fixture --repeats 1
python -m agent.memory.recall_eval --provider fixture --repeats 1
python -m agent.hitl.eval --provider fixture --repeats 1
python -m agent.agent_eval --provider fixture --repeats 1
```

`agent.main` / `agent.http` 使用同一 model-first runtime。`python -m agent.model_eval` 是首轮决策 fixture eval（不执行研究），`agent.agent_eval` 保留旧 trace 的离线 diagnostic dataset；它不验证当前 native loop。远程 eval 不属于测试套件。

测试套件：

```bash
python -m pytest -q tests/test_agent_model_loop.py tests/test_agent_evidence.py tests/test_agent_observability.py tests/test_agent_safety.py
```

这是小型 runtime fast gate，不是全仓验证。默认 pytest collection 排除 `tests/sandbox/`；真实 isolation/resource tests 必须显式 opt-in，不能作为普通 coding validation。分类与命令见 [tests README](tests/README.md)。
