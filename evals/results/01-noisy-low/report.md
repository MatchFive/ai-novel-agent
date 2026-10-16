# 一致性审校评测报告 · 01-noisy-low

- 数据集：consistency_v1.jsonl
- provider：noisy
- strategy：single
- model：-

## 总指标

| 指标 | 值 | 说明 |
|---|---|---|
| 检出率 Recall | **86.67%** | 人工种的 bug 被找出来的比例（漏检＝读者出戏） |
| 必检出 Recall | **85.71%** | 仅统计 must_detect 标注的核心问题 |
| 严重度加权 Recall | 88.24% | critical×3 / major×2 / minor×1 |
| 准确率 Precision | **56.52%** | 报出的问题里真问题的比例（误报＝用户被噪音淹没） |
| F1 | 0.684 | |
| 类型判定准确率 | 76.92% | 命中里 type 判对的比例 |
| 干净章误报率 | **0.60 条/篇** | 5 篇干净/诱饵章上平均报出的假问题数 |

（gold 15 条 / 模型报出 23 条：TP=13 FP=10 FN=2）

## 分类型表现

| 类型 | gold | 命中 | 检出率 | 报出 | 类型判定准确率 |
|---|---|---|---|---|---|
| timeline | 2 | 2 | 100% | 4 | 50% |
| setting | 2 | 2 | 100% | 7 | 100% |
| character | 2 | 1 | 50% | 3 | 100% |
| memory_conflict | 2 | 1 | 50% | 1 | 100% |
| knowledge_leak | 2 | 2 | 100% | 2 | 50% |
| promise_break | 2 | 2 | 100% | 2 | 100% |
| ai_flavor | 3 | 3 | 100% | 4 | 67% |

## 逐条明细

| case | 类别 | 难度 | gold | 报出 | 命中 | 漏检 | 误报 | 类型错配 |
|---|---|---|---|---|---|---|---|---|
| tc01 | gold_bug | easy | 1 | 2 | 1 | 0 | 1 | timeline→ai_flavor |
| tc02 | gold_bug | medium | 1 | 1 | 1 | 0 | 0 | - |
| tc03 | gold_bug | easy | 1 | 2 | 1 | 0 | 1 | - |
| tc04 | gold_bug | medium | 1 | 1 | 1 | 0 | 0 | - |
| tc05 | gold_bug | easy | 1 | 1 | 0 | 1 | 1 | - |
| tc06 | gold_bug | medium | 1 | 2 | 1 | 0 | 1 | - |
| tc07 | gold_bug | easy | 1 | 0 | 0 | 1 | 0 | - |
| tc08 | gold_bug | medium | 1 | 1 | 1 | 0 | 0 | - |
| tc09 | gold_bug | easy | 1 | 1 | 1 | 0 | 0 | knowledge_leak→timeline |
| tc10 | gold_bug | hard | 1 | 1 | 1 | 0 | 0 | - |
| tc11 | gold_bug | medium | 1 | 2 | 1 | 0 | 1 | - |
| tc12 | gold_bug | hard | 1 | 1 | 1 | 0 | 0 | - |
| tc13 | gold_bug | easy | 1 | 1 | 1 | 0 | 0 | ai_flavor→knowledge_leak |
| tc14 | gold_bug | medium | 1 | 3 | 1 | 0 | 2 | - |
| tc15 | gold_bug | hard | 1 | 1 | 1 | 0 | 0 | - |
| tc16 | clean | medium | 0 | 0 | 0 | 0 | 0 | - |
| tc17 | clean | hard | 0 | 1 | 0 | 0 | 1 | - |
| tc18 | clean | hard | 0 | 1 | 0 | 0 | 1 | - |
| tc19 | decoy | hard | 0 | 1 | 0 | 0 | 1 | - |
| tc20 | decoy | hard | 0 | 0 | 0 | 0 | 0 | - |

### 漏检的 gold

- `tc05` → 沈砚哈哈大笑，一巴掌拍在陆知雨肩上
- `tc07` → 沈砚第一次动用霜纹

### 误报（Top 15）

- `tc01` [character] 她笑了笑 — 疑似腔调不符
- `tc03` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
- `tc05` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
- `tc06` [timeline] 第二天 — 疑似时间跳跃
- `tc11` [ai_flavor] 他抬起头 — 疑似套话）
- `tc14` [character] 她笑了笑 — 疑似腔调不符
- `tc14` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
- `tc17` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
- `tc18` [timeline] 第二天 — 疑似时间跳跃
- `tc19` [setting] 他从怀里掏出一柄铁剑抵住门框 — 疑似与设定冲突（原文不存在）
