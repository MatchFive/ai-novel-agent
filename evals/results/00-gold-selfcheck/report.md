# 一致性审校评测报告 · 00-gold-selfcheck

- 数据集：consistency_v1.jsonl
- provider：gold
- strategy：single
- model：-

## 总指标

| 指标 | 值 | 说明 |
|---|---|---|
| 检出率 Recall | **100.00%** | 人工种的 bug 被找出来的比例（漏检＝读者出戏） |
| 必检出 Recall | **100.00%** | 仅统计 must_detect 标注的核心问题 |
| 严重度加权 Recall | 100.00% | critical×3 / major×2 / minor×1 |
| 准确率 Precision | **100.00%** | 报出的问题里真问题的比例（误报＝用户被噪音淹没） |
| F1 | 1.000 | |
| 类型判定准确率 | 100.00% | 命中里 type 判对的比例 |
| 干净章误报率 | **0.00 条/篇** | 5 篇干净/诱饵章上平均报出的假问题数 |

（gold 15 条 / 模型报出 15 条：TP=15 FP=0 FN=0）

## 分类型表现

| 类型 | gold | 命中 | 检出率 | 报出 | 类型判定准确率 |
|---|---|---|---|---|---|
| timeline | 2 | 2 | 100% | 2 | 100% |
| setting | 2 | 2 | 100% | 2 | 100% |
| character | 2 | 2 | 100% | 2 | 100% |
| memory_conflict | 2 | 2 | 100% | 2 | 100% |
| knowledge_leak | 2 | 2 | 100% | 2 | 100% |
| promise_break | 2 | 2 | 100% | 2 | 100% |
| ai_flavor | 3 | 3 | 100% | 3 | 100% |

## 逐条明细

| case | 类别 | 难度 | gold | 报出 | 命中 | 漏检 | 误报 | 类型错配 |
|---|---|---|---|---|---|---|---|---|
| tc01 | gold_bug | easy | 1 | 1 | 1 | 0 | 0 | - |
| tc02 | gold_bug | medium | 1 | 1 | 1 | 0 | 0 | - |
| tc03 | gold_bug | easy | 1 | 1 | 1 | 0 | 0 | - |
| tc04 | gold_bug | medium | 1 | 1 | 1 | 0 | 0 | - |
| tc05 | gold_bug | easy | 1 | 1 | 1 | 0 | 0 | - |
| tc06 | gold_bug | medium | 1 | 1 | 1 | 0 | 0 | - |
| tc07 | gold_bug | easy | 1 | 1 | 1 | 0 | 0 | - |
| tc08 | gold_bug | medium | 1 | 1 | 1 | 0 | 0 | - |
| tc09 | gold_bug | easy | 1 | 1 | 1 | 0 | 0 | - |
| tc10 | gold_bug | hard | 1 | 1 | 1 | 0 | 0 | - |
| tc11 | gold_bug | medium | 1 | 1 | 1 | 0 | 0 | - |
| tc12 | gold_bug | hard | 1 | 1 | 1 | 0 | 0 | - |
| tc13 | gold_bug | easy | 1 | 1 | 1 | 0 | 0 | - |
| tc14 | gold_bug | medium | 1 | 1 | 1 | 0 | 0 | - |
| tc15 | gold_bug | hard | 1 | 1 | 1 | 0 | 0 | - |
| tc16 | clean | medium | 0 | 0 | 0 | 0 | 0 | - |
| tc17 | clean | hard | 0 | 0 | 0 | 0 | 0 | - |
| tc18 | clean | hard | 0 | 0 | 0 | 0 | 0 | - |
| tc19 | decoy | hard | 0 | 0 | 0 | 0 | 0 | - |
| tc20 | decoy | hard | 0 | 0 | 0 | 0 | 0 | - |
