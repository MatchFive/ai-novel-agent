# evals · 一致性审校评测套件

> 归属：AI 小说创作云项目的**一致性审校**模块；本仓库为其「架构与评测展示层」，
> 本套件可直接独立运行（详见下方「独立运行」小节）。
> 被测对象：一致性审校任务 `task.consistency_scan`（生产实现中的审校服务）
> 目标：把「AI 审校到底准不准」从主观感受变成可复现的数字，并让每次改 prompt 都有据可依。

---

## 0. 这个套件解决什么问题

线上真实事故：同一章累积到 57 条 pending 问题，用户看到「错误越来越多、也没被标记解决」。
两种失败模式同时存在，而它们互相拉扯：

| 失败模式 | 用户感受 | 该盯的指标 |
|---|---|---|
| **漏检**（真问题没报） | 「AI 根本没看出来」→ 失去信任 | Recall |
| **误报**（没问题硬报） | 「全是废话，我都懒得看」→ 放弃使用 | Precision / 干净章误报率 |

只看其中一个必然失衡：把 prompt 写得越"警觉"，召回越高、误报越多。
所以评测必须**同时**给出两个方向的数字，并且有一个**干净章/诱饵章**子集专门测误报。

---

## 1. 文件结构

```
evals/
├── README.md                     ← 本文（完整路线）
├── datasets/
│   ├── build_dataset.py          ← 评测集生成脚本（case 写在这里，可评审、可 diff）
│   ├── context_base.json         ← 共享世界上下文（5 个变量与 _scan_context() 同名）
│   └── consistency_v1.jsonl      ← 20 条 case（生成产物）
├── metrics.py                    ← 匹配算法 + 指标（纯函数，无 LLM 依赖，可单测）
├── judge.py                      ← LLM-as-judge（匹配兜底 + 标注质检），无 Key 可离线跑
├── run_eval.py                   ← 本地 runner（三种 provider × 三种 strategy）
├── langsmith_eval.py             ← LangSmith 数据集上传 + 实验对比
├── results/                      ← 每次运行的 predictions / metrics / report
│   └── summary.csv               ← 横向对比表（每次运行追加一行）
└── requirements-eval.txt
```

**设计原则**：`metrics.py` 与 `judge.py` 不依赖 LLM、不依赖项目运行时，
所以它既能被 `run_eval.py` 调，也能被 `langsmith_eval.py` 调，还能进 pytest——
**同一套口径，三处复用**，避免"本地分数 0.8、平台分数 0.6"这种自欺。

---

## 2. 快速跑通（10 分钟）

```bash
cd evals
pip install -r requirements-eval.txt          # 只有 langsmith，本地评测零依赖

# ① 生成评测集
python datasets/build_dataset.py

# ② 自检 A：gold 当预测 → 指标必须全 1.0（验证"尺子"本身没坏）
python run_eval.py --provider gold

# ③ 自检 B：注入漏检/误报 → 指标必须按预期方向变（验证"尺子"是敏感的）
python run_eval.py --provider noisy --drop 0.15 --noise 0.2  --tag sanity-low
python run_eval.py --provider noisy --drop 0.40 --noise 0.7 --type-error 0.35 --tag sanity-high

# ④ 真跑基线（需要 OpenAI 兼容 Key）
export EVAL_API_KEY=sk-xxx
python run_eval.py --provider api --model gpt-4o-mini --tag baseline-single
python run_eval.py --provider api --model gpt-4o-mini --strategy self_consistency --repeat 3
python run_eval.py --provider api --model gpt-4o-mini --strategy verify --tag baseline-verify
cat results/summary.csv
```

**已实测的自检结果**（本地已跑通，可作为回归基线）：

| 运行 | Recall | 必检出 Recall | Precision | F1 | 类型准确率 | 干净章误报率 |
|---|---|---|---|---|---|---|
| gold（自检上限） | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 |
| noisy 低（drop .15 / noise .2） | 0.867 | 0.857 | 0.565 | 0.684 | 0.769 | 0.60 |
| noisy 高（drop .40 / noise .70） | 0.600 | 0.643 | 0.273 | 0.375 | 0.222 | 0.80 |

失败模式不同的注入让指标沿不同方向移动，说明各维度**没有被互相污染**。

> **实测发现（值得记住的一条）**：`--use-judge` 在 noisy 模式下对分数**没有任何影响**。
> 原因是噪声数据直接复用了 gold 的原文 quote，L1 锚点已经命中，裁判根本不会被触发。
> 裁判的价值只体现在**真实模型输出**上——它会用不同措辞概括同一处矛盾、或只抄半句原文。
> 所以要验证裁判是否有效，必须拿真实 run 的输出做对照：
> `python run_eval.py --provider api --tag real-nojudge` vs `... --tag real-judge --use-judge`。
> 这个"指标动了但没动在预期的地方 → 先怀疑数据而不是模型"的排查习惯，本身就是评测工作的一半。

---

## 3. 评测集设计（20 条）

### 3.1 分布

| 类别 | 条数 | 作用 |
|---|---|---|
| 人工种 bug | 15 | 7 类问题各 2–3 条，测 **Recall**（timeline 2 / setting 2 / character 2 / memory_conflict 2 / knowledge_leak 2 / promise_break 2 / ai_flavor 3） |
| 干净章 clean | 3 | 全无问题，测 **Precision**（tc17/tc18 还刻意埋了「看似可疑实则合规」的细节） |
| 诱饵 decoy | 2 | **看似违规、实则合法**，专测抗误报（tc19 像知识泄漏、tc20 像时间线冲突） |

世界设定：《霜降之城》——同一套铁律/角色卡/记忆/知识状态/承诺，20 条共用。
好处：① 考验真正的"跨文档对照"能力，而不是单篇阅读理解；② 上下文可复用，数据集体积小。

### 3.2 三个关键设计决策

**① 造诱饵章，比造 bug 章更重要。**
只种 bug 的评测集，会把"什么都报一遍"的模型评成满分。
`tc19`（陆知雨说"我知道你不是普通人"）和 `tc20`（双线交叉叙述的时间差）是全文最难的两条——
它们检验的是模型**认不认得"合法"**。诱饵章上的误报必须单独看，它比整体 precision 更能说明问题。

**② 多标签容错：允许类型判定有争议。**
`tc02`（时间推算错）既可判 `timeline` 也可判 `memory_conflict`；`tc12` 既可判 `promise_break`
也可判 `memory_conflict`。这类 case 在 `gold_issues` 里带 `acceptable_types` 字段，
命中任一即算类型正确。**评测集要容忍真实存在的标注歧义，否则指标会奖励"猜标注者偏好"。**

**③ 用 `must_detect` 区分"必须抓住"和"加分项"。**
`tc12` 标了 `must_detect=False`：它是类型边界案例，抓不到不扣必检出分，
但会体现在总 Recall 里——让核心指标不被争议样本拖累。

### 3.3 标注规范（扩集时照这个来）

每条 gold 必填 4 项 + 1 可选：
- `type`：7 类之一（与提示词里的定义严格一致）
- `quote`：**必须是从 draft_text 里能精确复制的原文片段**（这是匹配锚点，错了整条就废）
- `severity`：critical / major / minor —— 用于严重度加权 Recall
- `note`：为什么算问题 / 依据哪条设定，写给人看，也是 judge 的参考
- `must_detect` / `acceptable_types`：见上

**避免自我偏心**：写完 gold 隔一天再复核一遍；或者用 `judge.audit_gold()` 让一个**不同厂商的模型**
当质检员抽查——它对"这条矛盾真的存在吗"的独立判断，能抓出你写错/写漏的标注。

---

## 4. 指标口径（每个指标替谁说话）

| 指标 | 定义 | 它替谁说话 |
|---|---|---|
| **Recall** | 命中的 gold / gold 总数 | 读者——漏掉一个真矛盾，人设就崩了 |
| **必检出 Recall** | 只统计 `must_detect` | 编辑——核心问题一个都不能漏 |
| **严重度加权 Recall** | critical×3 / major×2 / minor×1 | 编辑——把"抓大放小"的能力量化 |
| **Precision** | 命中 / 模型报出总数 | 作者——报一堆废话等于没报 |
| **干净章误报率** | 干净章上平均报出的假问题条数 | 作者——最直观的"烦人指数" |
| **类型判定准确率** | 命中里 type 一致的比例 | 产品——类型错了，后续自动修复就会用错策略 |
| **F1** | 上面两者的调和平均 | 汇报——一个能进周报的总分 |
| **稳定性** | 同 case 多次运行的 issue 集合一致性 | 工程——temperature 0.1 也会飘，飘多少？ |
| **成本 / 延迟** | token 数、p50 / max 延迟 | 老板——审校是高频调用，成本敏感 |

匹配算法两级（`metrics.match_issues`）：
- **L1 quote 锚点**：归一化（去空白/标点、全角转半角）后判断包含关系或 `SequenceMatcher` 相似度 ≥ 0.62
- **L2 judge 兜底**：L1 未匹配的对，交给裁判模型判"是否同一处问题"
- **贪心一对一**：按相似度降序认领，防止一条 gold 被多条预测重复认领（虚高 Recall）

---

## 5. 实验矩阵：跑出你的"数据驱动"故事

按顺序跑，每一步都回答一个具体问题：

| # | 实验 | 目的 | 看什么 |
|---|---|---|---|
| E0 | provider=gold | 尺子自检 | 必须全 1.0，否则先修指标代码 |
| E1 | provider=noisy ×2 档 | 尺子灵敏度 | 指标应随噪声单调恶化 |
| E2 | api + single | 现状基线 | 记下 Recall / Precision / 干净章误报率 |
| E3 | api + self_consistency (repeat=3) | 多数票能否压误报 | Precision ↑？Recall 掉多少？成本 ×3 换来的值不值 |
| E4 | api + verify | 两阶段（扫描→裁决） | Precision 提升幅度 vs 额外一次调用成本 |
| E5 | 改 prompt 后重跑 E2 | prompt 迭代是否真的有效 | 与 baseline 逐条 diff，看**修好了什么、改坏了什么** |
| E6 | 换模型（lite vs pro 档） | 该任务值不值 pro 档 | 分数差 / 成本差，给路由表一个依据 |

E5 是核心：**改 prompt 不再是"感觉好一点了"，而是"Recall +6%、误报 -0.4 条/篇、tc15 依然漏"**。
逐条 diff 还能发现"修好 tc09、弄坏 tc19"这类此消彼长——这正是评测集的价值。

---

## 6. 接 LangSmith（可选，但强烈建议做一轮）

```bash
pip install langsmith
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_TRACING=true
export EVAL_API_KEY=sk-xxx

python langsmith_eval.py --upload-only                  # 上传 20 条 dataset
python langsmith_eval.py --model gpt-4o-mini            # 跑实验 A
python langsmith_eval.py --strategy self_consistency --repeat 3   # 跑实验 B
# 到 smith.langchain.com 的 Experiments 页并排对比 A / B
```

上传的 each example 里带着**渲染好的完整 prompt**，所以在平台上点开任意一条 run，
能直接看到模型当时看到了什么上下文——查"为什么这条漏了"时省掉大量复现工作。

---

## 7. 接 CI（把评测变成门禁）

`.github/workflows/eval.yml`（示意，需在仓库 Secrets 配 `EVAL_API_KEY`）：

```yaml
name: eval-consistency
on:
  pull_request:
    paths: ["evals/**", "prompts/**"]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r evals/requirements-eval.txt
      - name: 尺子自检（无 Key，必须过）
        run: |
          cd evals
          python datasets/build_dataset.py
          python run_eval.py --provider gold --tag ci-selfcheck
      # 有 Key 时再跑真评测，并用阈值卡住回归
      - name: 真评测
        if: env.EVAL_API_KEY != ''
        env: { EVAL_API_KEY: "${{ secrets.EVAL_API_KEY }}" }
        run: cd evals && python run_eval.py --provider api --model ${{ vars.EVAL_MODEL }} --tag ci-real
```

门禁阈值建议（先用基线分数 - 容差，别一上来就卡死）：
`must_detect_recall ≥ 基线 - 0.05` 且 `干净章误报率 ≤ 基线 + 0.3`。

---

## 8. 面试怎么讲（质量闭环）

**一句话版本**：
> 「我给 AI 审校建了一个 20 条的评测集，其中 5 条是**干净章和诱饵章**——专门测误报。
> 用 quote 锚点匹配 gold，跑出 Recall / Precision / 干净章误报率三个数，
> 把"改 prompt"从感觉变成 +6% 召回、-0.4 条误报。再把这套尺子接进 CI 当门禁。」

**墙上要能画出的闭环（7 步）**：

```
        ┌────────────────────────────────────────────────┐
        │  ① 线上真实失败样本（用户反馈 / 低分 run 回流） │
        └───────────────────────┬────────────────────────┘
                                ↓ 脱敏 + 标注
        ┌────────────────────────────────────┐
        │  ② 评测集（20+ 条，含干净章/诱饵章）│←── ③ 裁判模型质检标注
        └───────────────────────┬────────────┘
                                ↓
        ┌────────────────────────────────────┐
        │  ④ 跑评测：quote 锚点 + judge 兜底  │
        └───────────────────────┬────────────┘
                                ↓
        ┌────────────────────────────────────┐
        │  ⑤ 指标：Recall / Precision / 误报率 │
        │      / 类型准确率 / 成本 / 延迟      │
        └───────────────────────┬────────────┘
                                ↓ 定位失败 case（逐条 diff）
        ┌────────────────────────────────────┐
        │  ⑥ 改 prompt / 改架构（多数票、验证器）│
        └───────────────────────┬────────────┘
                                ↓ 回归对比，分数不降才合并
        ┌────────────────────────────────────┐
        │  ⑦ CI 门禁 + LangSmith 留痕 → 回到 ① │
        └────────────────────────────────────┘
```

**必被追问的 6 个问题，准备好答案**：

1. **就 20 条，能说明什么？**
   → 20 条不足以证明"绝对准确率"，但足以稳定地做**相对比较**（A/B prompt、换模型、换策略）。
   指标口径固定后，方差主要来自模型采样而非样本量，所以相对结论可靠；扩集是加法，不是前提。
2. **gold 是你自己写的，不偏吗？**
   → 有缓解措施：quote 必须可从正文精确复制（防止含糊标注）；多标签容错（承认争议）；
   隔日复核 + 换一个不同厂商的模型做标注质检（`audit_gold`）。
3. **为什么用 quote 匹配而不是语义相似度？**
   → quote 是**取证锚点**：审校问题的价值一半在于"指得准"，指不到具体句子用户没法改。
   语义相似度会把"同一类但不同处的两个问题"错认成一命中，虚高召回。语义只做 L2 兜底。
4. **为什么不用 BLEU/ROUGE 之类的通用指标？**
   → 生成任务里它们是弱代理；这里是**检测任务**，本质是信息检索，precision/recall 才是对的框架。
5. **误报和漏检哪个更严重？怎么权衡？**
   → 产品上误报更致命（真实事故：57 条 pending 导致用户弃用），所以门禁把干净章误报率作为硬约束；
   但 must_detect 的核心问题一个都不许漏。策略上用「先召回后精准」两阶段来同时压两边。
6. **这套东西在别的 Agent 上能复用吗？**
   → 能。骨架是「失败样本 → 标注集 → 多指标 → 迭代 → 门禁」，
   换个任务只需换 gold 的定义与匹配器（生成类任务换成 rubric 打分 + LLM judge 的一致性校验，
   比如本项目的正文质量评测。这是同一个闭环。）

---

## 9. 时间预算与下一步

| 阶段 | 内容 | 用时 |
|---|---|---|
| D1 | 跑通自检（gold / noisy），读懂指标口径 | 半天 |
| D1 | 扩集到 30–40 条（重点补 ai_flavor 与诱饵章） | 半天 |
| D2 | 用真模型跑 E2–E4 基线，记进 results/summary.csv | 半天 |
| D3 | 改 prompt 跑 E5，写《prompt 迭代记录》小节（含逐条 diff） | 1 天 |
| D4 | 接 LangSmith 上传 + 两轮实验对比 | 半天 |
| D5 | 接 GitHub Actions 门禁 + 写 README 面试话术 | 半天 |

**数据飞轮（长期）**：把线上 `revision_issues` 里被用户标 `ignored`（判为误报）和
被人工额外补充的问题**回流**进评测集——线上每出现一类新失败，评测集就厚一层。
这是从"静态评测集"升级到"持续学习闭环"的关键一步，也是面试里最能拉开差距的一句话。

---

## 10. 与项目的关系

- 评测覆盖的 `task.consistency_scan` 与线上同一条模板（生产环境从项目的提示词文件读取；
  展示层仓库不含该文件，会回退到结构等价的精简模板，见 `run_eval.py` 的 `PROMPTS_JSON`），
  **不另写一份 prompt**，保证"评的就是跑的"。
- 本目录是纯增量：不 import 项目运行时、不连数据库、不改现有代码，
  可独立运行，也可后续接进项目的测试目录或 CI。
- 若要把评测结果反哺产品：`revision_issues` 已有 `scan_id` 维度，
  后续可按 scan_id 统计线上误报率，与评测集分数相互校验。
