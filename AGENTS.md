# Tradar Agent 工作指南

Tradar 是一个量化研究、策略验证与生产信号平台，并正在逐步建设为 Agentic Quantitative Research Platform。

本文件用于帮助 coding / research agent 在收到新的研究任务后，快速理解项目、复用已有能力、执行研究并向用户反馈结果。

## 1. 收到任务后先做什么

用户通常只会给出一句研究目标，例如：

> 研究 A 股相对指数的 52 周动量是否具有预测能力。

不要立即从零写实现，也不要先设计新框架。

首先理解研究问题，并快速检查现有代码：

1. 查看 `research/`，确认是否已有可复用的研究能力。
2. 查看 `scripts/` 中是否已有相似研究、因子、策略或数据处理脚本。
3. 必要时查看 `utils/`，确认已有的数据加载接口。
4. 如果任务已经对应 `agent/` 中存在的 Agent Tool，优先使用稳定 Tool contract，而不是重新实现同样逻辑。

优先复用已有能力。只有现有能力确实不足时，才新增最小实现。

## 2. 项目边界

策略研究目录职责：

- `agent/core/`：共享 Tool contracts、ResearchRun、provider adapters 与资源加载。
- `agent/tools/`：当前真实存在的 Tool Calling capability、production tools 与其 eval runner。
- `agent/context/`：Context selection 与 Context Engineering eval runner。
- `research/`：唯一 canonical quantitative research engine。
- `scripts/`：具体研究实验与一次性研究入口。
- `utils/`：数据加载与通用基础设施。
- `resources/`：非执行型 eval dataset、稳定 prompt 与知识记录约定，不放 Python 代码或私有历史。
- `tests/`：关键公共能力的回归测试。

依赖关系应尽量保持：`agent → research → utils` 。Agent 层负责理解任务和调用能力，`research/` 负责确定性的量化计算。不要把特定研究实验写进 `research/`。实验性逻辑通常应该放在 `scripts/`。


## 3. 如何开展新的量化研究

对于新的研究问题，默认采用以下工作流：

### 第一步：把问题转成可验证假设

明确：

- 研究对象 / universe
- 因子或事件定义
- benchmark
- 观察区间
- 预测 horizon
- 评价指标
- 是否需要组合回测

不要一开始就做大量参数扫描。如果问题首先是在问“这个因子有没有预测能力”，先做 IC、分组收益、年度稳定性等 factor evaluation，再决定是否值得进入策略回测。不要为了得到漂亮结果而直接跳到组合参数优化。

### 第二步：复用 `research/`

常用能力优先从现有模块寻找，例如：

- universe / tradability：`research.panel`
- factor evaluation：`research.factor_analyzer`、`research.factor_dsl`
- performance metrics：`research.metrics`
- backtest：`research.vector_backtest`
- result comparison：`research.report`
- risk overlay：`research.risk_overlay`

先阅读真实实现和函数签名，再调用。不要仅根据文件名或 README 猜测函数行为。

### 第三步：研究代码放入 `scripts/`

新的研究实验默认写到：

```text
scripts/<research_topic>/
```

同一个研究主题下产生的多个脚本放在同一个子目录，不要持续平铺到 `scripts/` 根目录。

例如：

```text
scripts/relative_momentum/
    factor_eval.py
    backtest.py
```

数据下载、通用维护等跨研究脚本可以继续放在 `scripts/` 根目录。

研究脚本应该尽量薄：

`加载数据 → 调用 research 能力 → 整理结果 → 输出`

不要在脚本里重新实现一套 backtest、factor analyzer 或 universe engine。

## 4. 如何运行研究

项目通常从 repo root 执行。

研究脚本应支持：

```bash
python -m scripts.<topic>.<script>
```

如果需要项目路径：

```bash
PYTHONPATH=. python -m scripts.<topic>.<script>
```

先打印研究版本、关键参数和样本范围，再执行耗时的数据加载或计算，方便确认当前运行的到底是哪一版实验。研究必须真实执行后再汇报结果。不要根据代码逻辑推测一个“应该大概是多少”的结果。

## 5. 研究结果怎么输出

终端结果应以紧凑 DataFrame 为主，避免大量散乱日志。表格默认使用类似：`df.to_string(index=False, col_space=8)`。策略表现指标尽量放在同一张 metrics 表中，不要把 return metrics 和 execution metrics 拆成很多小表。例如：

- Ann
- Cum
- Active
- Sharpe
- MDD
- TO
- Cost
- Gross


研究脚本只负责输出事实和指标，不要在脚本中打印诸如：

`Decision: PASS`
`Recommendation: Launch`

研究判断由 Agent 在看到真实结果后完成。

## 6. 如何向用户反馈

完成研究后，反馈重点不是“我写了哪些代码”，而是研究结论。默认按照下面的顺序：

1. **研究了什么**
   - 假设、样本、benchmark、horizon、主要方法。

2. **实际结果**
   - 关键 IC、收益、Sharpe、MDD、turnover、cost 或其它真正相关指标。

3. **结果说明什么**
   - 是否存在稳定预测能力。
   - Alpha 来自哪里。
   - 哪些阶段有效 / 失效。
   - 是否值得进入下一步。

4. **异常或限制**
   - 数据覆盖、PIT、tradability、benchmark alignment 等真正影响结论的问题。

5. **下一步**
   - Reject。
   - 继续研究一个明确机制。
   - 进入组合回测。
   - 形成 strategy candidate。

不要因为结果不好就自动继续增加十几个 validation。如果机制和结论已经足够清楚，应停止低价值验证并直接给出结论。

## 7. 研究完整性原则

以下规则属于 Tradar 对于A股的的核心研究口径。

### A股T+1规则

`research.vector_backtest` 是 canonical backtest engine。它内部已经处理 T+1：`signal → next trading day execution`调用方不要再次对 signal 做 `shift(1)`，否则会变成 T+2。

### 交易成本

买卖成本分开计算。默认研究口径：

- buy cost：10 bp
- sell cost：15 bp

除非具体实验明确说明，否则不要随意覆盖成其它成本。

### Tradability

优先使用 `research.panel` 中已有的：

- eligible universe
- buyable
- sellable
- trading status
- price-limit rules

不要在研究脚本里自行用简单的 `±9.5%` 等规则重新判断涨跌停。

### PIT / Look-ahead

财务数据、因子数据和 forward return 必须保持时间因果关系。未来收益只能作为 label / evaluation target，不能进入当期 factor construction。如果某项数据是否 PIT-safe 无法确认，应明确报告，而不是默认认为安全。

### Benchmark

相对收益、active return、relative momentum 等研究必须明确 benchmark，并保证日期正确对齐。不要把 benchmark 缺失或错位后的结果当成有效 alpha。

## 8. Agent Tool 的定位

Tradar 正在逐步把原本依赖研究人员和脚本完成的流程显式化为 Agent：

```text
用户问题
→ Agent 理解与规划
→ 选择 Tool
→ 调用 research 确定性能力
→ 获取结构化结果
→ Validation
→ Interpretation
→ Research Memory
→ Human Review
→ Production
```

`agent/tools/` 中的 Tool 是稳定的业务能力 contract，而不是把 `research/` 中每个 Python 函数机械地暴露成 Tool。

当前和计划中的核心 Tool 包括：

- `inspect_universe`
- `evaluate_factor`
- `run_backtest`
- `validate_research_run`
- `compare_runs`

后续还会增加 research retrieval / memory、event study 和 strategy candidate 等能力。如果已有 Agent Tool 能完整覆盖任务，优先调用 Tool。如果当前 Tool 还不能覆盖新的探索性研究，可以继续采用：`研究问题 → scripts 实验 → research engine` 研究成熟后，再考虑是否值得沉淀为新的稳定 Agent Tool。不要为了“Agent 化”而把所有实验代码都包装成 Tool。

当前 capability eval runner：

```bash
python -m agent.tools.eval --provider fixture --repeats 1
python -m agent.context.eval --provider fixture --repeats 1
```

`resources/eval/` 中的 JSON 是 dataset，`agent/*/eval.py` 是 runner，`agent/tools/tools.py` 是 production capability。三者保持分离；远程 DeepSeek/OpenAI eval 由调用者自行运行。

## 9. 修改公共研究能力时

一般研究任务优先新增或修改 `scripts/`。只有在确认缺少的是通用能力时，才修改 `research/`。修改 `research/` 前应确认：

- 是否已经存在类似实现；
- 是否会改变现有研究口径；
- 是否影响 T+1、PIT、transaction cost、tradability 等核心语义；
- 是否应该成为所有研究共享的能力。

不要因为单个实验方便，就改变 canonical engine 的计算语义。

## 10. 总原则

Tradar 的研究工作优先级是：

**真实研究问题 > 复用现有能力 > 最小实验 > 真实执行 > 结果解释 > 必要验证 > 产品化**

避免：

- 为一个实验设计新框架；
- 重复实现已有 research engine；
- 无意义参数扫描；
- 为验证而验证；
- 只写代码不运行；
- 根据未执行代码编造结果；
- 把探索性脚本过早产品化；
- 为了使用 Agent 技术而引入不必要的 Agent 基础设施。

最终目标不是让 Agent 写更多代码，而是让一次量化研究能够被稳定地：

**理解、执行、验证、解释、记录和复用。**
