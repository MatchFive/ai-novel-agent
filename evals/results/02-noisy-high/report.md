# 一致性审校评测报告 · 02-noisy-high

- 数据集：consistency_v1.jsonl
- provider：noisy
- strategy：single
- model：-

## 总指标

| 指标 | 值 | 说明 |
|---|---|---|
| 检出率 Recall | **60.00%** | 人工种的 bug 被找出来的比例（漏检＝读者出戏） |
| 必检出 Recall | **64.29%** | 仅统计 must_detect 标注的核心问题 |
| 严重度加权 Recall | 61.76% | critical×3 / major×2 / minor×1 |
| 准确率 Precision | **27.27%** | 报出的问题里真问题的比例（误报＝用户被噪音淹没） |
| F1 | 0.375 | |
| 类型判定准确率 | 22.22% | 命中里 type 判对的比例 |
| 干净章误报率 | **0.80 条/篇** | 5 篇干净/诱饵章上平均报出的假问题数 |

（gold 15 条 / 模型报出 33 条：TP=9 FP=24 FN=6）

## 分类型表现

| 类型 | gold | 命中 | 检出率 | 报出 | 类型判定准确率 |
|---|---|---|---|---|---|
| timeline | 2 | 1 | 50% | 7 | 0% |
| setting | 2 | 1 | 50% | 9 | 100% |
| character | 2 | 2 | 100% | 5 | 0% |
| memory_conflict | 2 | 2 | 100% | 0 | 0% |
| knowledge_leak | 2 | 1 | 50% | 1 | 0% |
| promise_break | 2 | 0 | 0% | 4 | 0% |
| ai_flavor | 3 | 2 | 67% | 7 | 50% |

## 逐条明细

| case | 类别 | 难度 | gold | 报出 | 命中 | 漏检 | 误报 | 类型错配 |
|---|---|---|---|---|---|---|---|---|
| tc01 | gold_bug | easy | 1 | 2 | 0 | 1 | 2 | - |
| tc02 | gold_bug | medium | 1 | 3 | 1 | 0 | 2 | timeline→promise_break |
| tc03 | gold_bug | easy | 1 | 2 | 0 | 1 | 2 | - |
| tc04 | gold_bug | medium | 1 | 2 | 1 | 0 | 1 | - |
| tc05 | gold_bug | easy | 1 | 3 | 1 | 0 | 2 | character→promise_break |
| tc06 | gold_bug | medium | 1 | 2 | 1 | 0 | 1 | character→setting |
| tc07 | gold_bug | easy | 1 | 2 | 1 | 0 | 1 | memory_conflict→promise_break |
| tc08 | gold_bug | medium | 1 | 2 | 1 | 0 | 1 | memory_conflict→timeline |
| tc09 | gold_bug | easy | 1 | 1 | 1 | 0 | 0 | knowledge_leak→promise_break |
| tc10 | gold_bug | hard | 1 | 2 | 0 | 1 | 2 | - |
| tc11 | gold_bug | medium | 1 | 1 | 0 | 1 | 1 | - |
| tc12 | gold_bug | hard | 1 | 1 | 0 | 1 | 1 | - |
| tc13 | gold_bug | easy | 1 | 2 | 1 | 0 | 1 | - |
| tc14 | gold_bug | medium | 1 | 2 | 1 | 0 | 1 | ai_flavor→knowledge_leak |
| tc15 | gold_bug | hard | 1 | 2 | 0 | 1 | 2 | - |
| tc16 | clean | medium | 0 | 1 | 0 | 0 | 1 | - |
| tc17 | clean | hard | 0 | 1 | 0 | 0 | 1 | - |
| tc18 | clean | hard | 0 | 1 | 0 | 0 | 1 | - |
| tc19 | decoy | hard | 0 | 0 | 0 | 0 | 0 | - |
| tc20 | decoy | hard | 0 | 1 | 0 | 0 | 1 | - |

### 漏检的 gold

- `tc01` → 正午的阳光把霜城的屋顶照得发白
- `tc03` → 攥住生锈的铁门闩，掌心的霜纹骤然亮起
- `tc10` → 你是最早的那一批
- `tc11` → 我早就告诉过你了，在霜降那天
- `tc12` → 铜……能替铁？我从来没听说过
- `tc15` → 有些事，不必说破。有些路，只能一个人走。

### 误报（Top 15）

- `tc01` [character] 她笑了笑 — 疑似腔调不符
- `tc01` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
- `tc02` [character] 她笑了笑 — 疑似腔调不符
- `tc02` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
- `tc03` [ai_flavor] 他抬起头 — 疑似套话）
- `tc03` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
- `tc04` [ai_flavor] 他抬起头 — 疑似套话）
- `tc05` [ai_flavor] 他抬起头 — 疑似套话）
- `tc05` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
- `tc06` [character] 她笑了笑 — 疑似腔调不符
- `tc07` [ai_flavor] 他抬起头 — 疑似套话）
- `tc08` [timeline] 第二天 — 疑似时间跳跃
- `tc10` [ai_flavor] 他抬起头 — 疑似套话）
- `tc10` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
- `tc11` [timeline] 第二天 — 疑似时间跳跃
