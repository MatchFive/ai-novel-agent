"""一致性审校评测：匹配算法与指标计算（纯函数、无 LLM 依赖、可单测）。

匹配是这类评测最难的地方：模型报的 issue 是自然语言，同一问题可以有十种说法，
quote 也可能只截取半句。这里用两级匹配：

  L1  quote 锚点匹配 —— 归一化后判断 gold.quote 与 pred.quote 是否有包含关系或高重叠
                        （quote 是"取证锚点"，匹配它比匹配 issue 说明文本可靠得多）
  L2  语义兜底匹配   —— 可选注入 judge（LLM 裁判），对 L1 未匹配的组合做二次判定

匹配采用「相似度降序贪心一对一」：避免一条 gold 被多条预测重复认领（会虚高 recall），
也避免一条预测同时认领多条 gold（会虚高 precision）。
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# 与一致性审校任务的提示词中点名的七类问题一致
ISSUE_TYPES = ["timeline", "setting", "character", "memory_conflict",
               "knowledge_leak", "promise_break", "ai_flavor"]

QUOTE_MATCH_THRESHOLD = 0.62   # quote 归一化后的相似度阈值
SEMANTIC_THRESHOLD = 0.60      # 兜底语义匹配阈值（judge 分数）


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def norm_text(s: str | None) -> str:
    """去掉空白与标点差异，全角转半角，便于中文片段比对。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。！？；：、,.!?;:\"'“”‘’（）()【】\[\]—…·\-]", "", s)
    return s


def quote_similarity(a: str | None, b: str | None) -> float:
    na, nb = norm_text(a), norm_text(b)
    if not na or not nb:
        return 0.0
    # 短片段被长片段完整包含 → 视为强匹配（模型常只抄半句）
    if len(na) >= 4 and na in nb:
        return 0.95
    if len(nb) >= 4 and nb in na:
        return 0.95
    return SequenceMatcher(None, na, nb).ratio()


# ---------------------------------------------------------------------------
# 两级匹配
# ---------------------------------------------------------------------------

def match_issues(pred_issues: list[dict], gold_issues: list[dict],
                 judge=None) -> dict:
    """返回 {pairs, unmatched_gold, unmatched_pred}。

    judge: 可选，callable(pred, gold) -> float（0~1 语义等价分）。传 None 则只做 L1。
    """
    preds = [p for p in pred_issues if isinstance(p, dict)]
    golds = list(gold_issues)

    candidates: list[tuple[float, int, int, str]] = []
    for gi, g in enumerate(golds):
        for pi, p in enumerate(preds):
            sim = quote_similarity(p.get("quote"), g.get("quote"))
            src = "quote"
            if sim < QUOTE_MATCH_THRESHOLD and judge is not None:
                j = judge(p, g)
                if j > sim:
                    sim, src = j, "judge"
            if sim >= SEMANTIC_THRESHOLD:
                candidates.append((sim, gi, pi, src))

    candidates.sort(key=lambda x: -x[0])
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[dict] = []
    for sim, gi, pi, src in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        g, p = golds[gi], preds[pi]
        pairs.append({
            "gold_index": gi,
            "pred_index": pi,
            "similarity": round(sim, 3),
            "matched_by": src,
            "gold_type": g.get("type"),
            "pred_type": p.get("type"),
            "type_ok": (p.get("type") == g.get("type")
                        or p.get("type") in (g.get("acceptable_types") or [])),
            "quote": g.get("quote"),
        })
    return {
        "pairs": pairs,
        "unmatched_gold": [golds[i] for i in range(len(golds)) if i not in used_g],
        "unmatched_pred": [preds[i] for i in range(len(preds)) if i not in used_p],
    }


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return round(a / b, 4) if b else 0.0


def compute_metrics(cases: list[dict], preds_by_id: dict[str, list[dict]],
                    judge=None, collect_details: bool = True,
                    skip_ids: set[str] | None = None) -> dict:
    """cases: 评测集（含 gold_issues）；preds_by_id: {case_id: [issue, ...]}

    skip_ids：调用失败、没有预测的 case。**必须排除出分母，不能当成「报出 0 条」**——
    落在干净章上会把一次超时算成「完美通过」，落在 bug 章上会把网络故障算成漏检，
    两个方向都在污染数字。排除后单独用 coverage 表达「这次评测覆盖了多少」。
    """
    tp = fp = fn = 0
    tp_must = fn_must = 0
    type_ok = 0
    weighted_hit = weighted_total = 0.0
    clean_cases = clean_fp = 0          # 干净章 / 诱饵章上的误报
    clean_fp_total = 0
    per_type = {t: {"gold": 0, "hit": 0, "reported": 0, "type_ok": 0} for t in ISSUE_TYPES}
    details: list[dict] = []
    skip_ids = skip_ids or set()
    errored: list[str] = []

    SEV_W = {"critical": 3, "major": 2, "minor": 1}

    for case in cases:
        cid = case["id"]
        if cid in skip_ids:
            errored.append(cid)
            if collect_details:
                details.append({
                    "id": cid, "category": case["category"], "difficulty": case["difficulty"],
                    "type_focus": case.get("type_focus"), "errored": True,
                    "gold": len(case.get("gold_issues") or []), "pred": 0, "hit": 0,
                    "missed": [], "false_positives": [], "type_mismatch": [],
                })
            continue
        golds = case.get("gold_issues") or []
        preds = preds_by_id.get(cid) or []
        r = match_issues(preds, golds, judge=judge)

        hits = len(r["pairs"])
        tp += hits
        fp += len(r["unmatched_pred"])
        fn += len(r["unmatched_gold"])
        type_ok += sum(1 for p in r["pairs"] if p["type_ok"])

        for g in golds:
            per_type.setdefault(g["type"], {"gold": 0, "hit": 0, "reported": 0, "type_ok": 0})
            per_type[g["type"]]["gold"] += 1
            w = SEV_W.get(g.get("severity", "major"), 2)
            weighted_total += w
            if g.get("must_detect"):
                pass
        for p in preds:
            t = p.get("type") or "unknown"
            per_type.setdefault(t, {"gold": 0, "hit": 0, "reported": 0, "type_ok": 0})
            per_type[t]["reported"] += 1
        for pair in r["pairs"]:
            per_type[pair["gold_type"]]["hit"] += 1
            if pair["type_ok"]:
                per_type[pair["gold_type"]]["type_ok"] += 1

        for gi, g in enumerate(golds):
            w = SEV_W.get(g.get("severity", "major"), 2)
            hit = any(p["gold_index"] == gi for p in r["pairs"])
            if hit:
                weighted_hit += w
            if g.get("must_detect"):
                if hit:
                    tp_must += 1
                else:
                    fn_must += 1

        if case["category"] in ("clean", "decoy"):
            clean_cases += 1
            clean_fp += len(r["unmatched_pred"])
            clean_fp_total += len(preds)

        if collect_details:
            details.append({
                "id": cid, "category": case["category"], "difficulty": case["difficulty"],
                "type_focus": case.get("type_focus"),
                "gold": len(golds), "pred": len(preds), "hit": hits,
                "missed": [g.get("quote") for g in r["unmatched_gold"]],
                "false_positives": [
                    {"type": p.get("type"), "quote": p.get("quote"), "issue": (p.get("issue") or "")[:80]}
                    for p in r["unmatched_pred"]
                ],
                "type_mismatch": [f"{p['gold_type']}→{p['pred_type']}" for p in r["pairs"] if not p["type_ok"]],
            })

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    must_recall = _safe_div(tp_must, tp_must + fn_must)

    return {
        "cases": len(cases),
        "covered_cases": len(cases) - len(errored),
        "errored_cases": len(errored),
        "errored_ids": errored,
        # 覆盖率：指标建立在多少比例的有效样本上。低于 1.0 时所有指标都只能当参考，
        # 因为失败的 case 既可能藏着漏检，也可能藏着误报。
        "coverage": _safe_div(len(cases) - len(errored), len(cases)),
        "gold_total": tp + fn,
        "pred_total": tp + fp,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "must_detect_recall": must_recall,
        "weighted_recall": _safe_div(weighted_hit, weighted_total),
        "type_accuracy": _safe_div(type_ok, tp),
        "clean_cases": clean_cases,
        "clean_false_positives": clean_fp,
        "clean_fp_rate": _safe_div(clean_fp, clean_cases),      # 平均每篇干净章报几条假问题
        "clean_fp_ratio": _safe_div(clean_fp_total and clean_fp or 0, clean_fp_total),
        "per_type": {
            t: {
                **v,
                "recall": _safe_div(v["hit"], v["gold"]),
                "type_accuracy": _safe_div(v["type_ok"], v["hit"]),
            } for t, v in per_type.items() if v["gold"] or v["reported"]
        },
        "details": details,
    }


def format_report(m: dict, title: str = "一致性审校评测报告", meta: dict | None = None) -> str:
    lines = [f"# {title}", ""]
    if meta:
        lines += [f"- {k}：{v}" for k, v in meta.items()] + [""]

    lines += ["## 总指标", "",
              "| 指标 | 值 | 说明 |", "|---|---|---|",
              f"| 检出率 Recall | **{m['recall']:.2%}** | 人工种的 bug 被找出来的比例（漏检＝读者出戏） |",
              f"| 必检出 Recall | **{m['must_detect_recall']:.2%}** | 仅统计 must_detect 标注的核心问题 |",
              f"| 严重度加权 Recall | {m['weighted_recall']:.2%} | critical×3 / major×2 / minor×1 |",
              f"| 准确率 Precision | **{m['precision']:.2%}** | 报出的问题里真问题的比例（误报＝用户被噪音淹没） |",
              f"| F1 | {m['f1']:.3f} | |",
              f"| 类型判定准确率 | {m['type_accuracy']:.2%} | 命中里 type 判对的比例 |",
              f"| 干净章误报率 | **{m['clean_fp_rate']:.2f} 条/篇** | {m['clean_cases']} 篇干净/诱饵章上平均报出的假问题数 |",
              f"| 覆盖率 | **{m['coverage']:.0%}** | {m['covered_cases']}/{m['cases']} 条 case 有有效预测"
              + (f"；{m['errored_cases']} 条未覆盖，已排除出全部分母" if m["errored_cases"] else "") + " |",
              "",
              f"（gold {m['gold_total']} 条 / 模型报出 {m['pred_total']} 条：TP={m['tp']} FP={m['fp']} FN={m['fn']}）",
              ""]

    if m["errored_cases"]:
        lines += [f"> ⚠️ 覆盖率 {m['coverage']:.0%}：{', '.join(f'`{i}`' for i in m['errored_ids'])} "
                  f"调用失败，未计入任何指标。这些 case 上真实表现未知，"
                  f"`clean_fp_rate` 等指标不能与全量运行直接比较。", ""]

    lines += ["## 分类型表现", "",
              "| 类型 | gold | 命中 | 检出率 | 报出 | 类型判定准确率 |", "|---|---|---|---|---|---|"]
    for t, v in m["per_type"].items():
        lines.append(f"| {t} | {v['gold']} | {v['hit']} | {v['recall']:.0%} | "
                     f"{v['reported']} | {v['type_accuracy']:.0%} |")

    if m.get("details"):
        lines += ["", "## 逐条明细", "",
                  "| case | 类别 | 难度 | gold | 报出 | 命中 | 漏检 | 误报 | 类型错配 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for d in m["details"]:
            if d.get("errored"):
                lines.append(f"| {d['id']} | {d['category']} | {d['difficulty']} | {d['gold']} | "
                             f"**失败** | - | - | - | 调用失败，未计入指标 |")
                continue
            lines.append(
                f"| {d['id']} | {d['category']} | {d['difficulty']} | {d['gold']} | {d['pred']} | "
                f"{d['hit']} | {len(d['missed'])} | {len(d['false_positives'])} | "
                f"{('; '.join(d['type_mismatch']) or '-')} |")

        missed = [d for d in m["details"] if d.get("missed")]
        if missed:
            lines += ["", "### 漏检的 gold", ""]
            for d in missed:
                for q in d["missed"]:
                    lines.append(f"- `{d['id']}` → {q}")

        fps = [(d["id"], fp) for d in m["details"] for fp in d["false_positives"]]
        if fps:
            lines += ["", "### 误报（Top 15）", ""]
            for cid, fp in fps[:15]:
                lines.append(f"- `{cid}` [{fp['type']}] {fp['quote']} — {fp['issue']}")
    return "\n".join(lines) + "\n"
