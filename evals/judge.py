"""LLM-as-judge：把"模型报的这条，和人工标的这条，是不是同一个问题"交给裁判模型。

为什么需要它：quote 匹配是字符级的，模型可能只抄了半句、或用不同措辞概括同一处矛盾。
只靠字符重叠会把「真命中」判成漏检，低估召回。judge 作为 L2 兜底（见 metrics.match_issues）。

两个用途：
  1. judge_pair(pred, gold) -> 0~1 等价分        —— 评测时兜底匹配
  2. audit_gold(case, gold_issue) -> {valid, reason} —— 元评测：检查人工标注本身写错没有
     （标注者也会偏心/看漏，这一步保证"标准答案"可信；面试聊到"评测集怎么保证质量"时用得上）

无 API Key 时用 heuristic judge（关键词 + 类型 + 共现），保证整套流程离线可跑。
"""
from __future__ import annotations

import json
import os
import re

from metrics import ISSUE_TYPES, norm_text, quote_similarity

PAIR_PROMPT = """你在做审校评测的判分。下面是【人工标注的问题】和【模型报出的问题】。
判断两者是否指向**同一处**问题（同一句原文、同一个矛盾），而不是两个碰巧相似但不同的地方。
注意：措辞不同、类型标签不同，只要指向的作用点一致就算同一处。
只输出 JSON：{{"same": true/false, "confidence": 0~1, "reason": "一句话"}}

【人工标注】type={g_type} quote={g_quote} 说明={g_note}
【模型报出】type={p_type} quote={p_quote} 说明={p_issue}
"""

AUDIT_PROMPT = """你是评测集质检员。下面是《霜降之城》的设定资料、一段正文，以及一条**人工标注的问题**。
请判断这条标注是否成立：它指出的矛盾/违规在资料与正文里真的存在吗？
只输出 JSON：{{"valid": true/false, "confidence": 0~1, "reason": "一句话"}}

【资料】
{context}

【正文】
{draft_text}

【人工标注】type={g_type} quote={g_quote} 说明={g_note}
"""

_STOP = set("的了吗呢是在和与或有也就都而及把被对从这那你我他她它们一个不没很".replace("\n", ""))


def _keywords(s: str) -> set[str]:
    s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", " ", str(s or ""))
    words = [w for w in re.split(r"\s+", s) if len(w) >= 2 and w not in _STOP]
    grams = {s[i:i + 2] for i in range(0, max(0, len(s) - 1))} if len(s) >= 4 else set()
    return set(words) | grams


def heuristic_judge(pred: dict, gold: dict) -> float:
    """离线兜底裁判：类型一致 + 说明文本有实质重叠时给较高分。"""
    type_ok = pred.get("type") == gold.get("type") or \
        pred.get("type") in (gold.get("acceptable_types") or [])
    overlap = 0.0
    kp = _keywords(f"{pred.get('issue', '')} {pred.get('quote', '')}")
    kg = _keywords(f"{gold.get('note', '')} {gold.get('quote', '')}")
    if kp and kg:
        overlap = len(kp & kg) / max(1, min(len(kp), len(kg)))
    base = 0.4 * overlap + (0.35 if type_ok else 0.0)
    base = max(base, quote_similarity(pred.get("quote"), gold.get("quote")) * 0.9)
    return round(min(base, 0.99), 3)


def build_llm_judge(base_url: str, api_key: str, model: str, temperature: float = 0.0):
    """返回 callable(pred, gold) -> float。调用失败时自动退回 heuristic。"""
    from run_eval import call_llm    # 延迟导入，避免循环依赖

    def judge(pred: dict, gold: dict) -> float:
        prompt = PAIR_PROMPT.format(
            g_type=gold.get("type"), g_quote=gold.get("quote"), g_note=gold.get("note", ""),
            p_type=pred.get("type"), p_quote=pred.get("quote"), p_issue=pred.get("issue", ""))
        try:
            text, _ = call_llm(prompt, base_url=base_url, api_key=api_key,
                               model=model, temperature=temperature, timeout=60)
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return heuristic_judge(pred, gold)
            data = json.loads(m.group(0))
            if not data.get("same"):
                return 0.0
            return float(data.get("confidence") or 0.8)
        except Exception:                    # noqa: BLE001
            return heuristic_judge(pred, gold)

    return judge


def audit_gold(case: dict, gold: dict, base_url: str, api_key: str, model: str) -> dict:
    """元评测：用裁判模型复核人工标注是否成立（抽查即可，20 条全过一遍也不贵）。"""
    from run_eval import call_llm

    ctx = dict(case["context"])
    ctx.update(case.get("context_overrides") or {})
    prompt = AUDIT_PROMPT.format(
        context="\n".join(f"【{k}】{v}" for k, v in ctx.items()),
        draft_text=case["draft_text"], g_type=gold.get("type"),
        g_quote=gold.get("quote"), g_note=gold.get("note", ""))
    text, _ = call_llm(prompt, base_url=base_url, api_key=api_key, model=model, temperature=0.0)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {"valid": None, "reason": "解析失败"}


def get_judge(args):
    """按 CLI 参数决定用真裁判还是离线兜底裁判。"""
    api_key = getattr(args, "api_key", None) or os.environ.get("EVAL_API_KEY") \
        or os.environ.get("OPENAI_API_KEY")
    if getattr(args, "use_judge", False) and api_key:
        jm = getattr(args, "judge_model", None) or args.model
        print(f"  裁判模型：{jm}")
        return build_llm_judge(args.base_url, api_key, jm)
    if getattr(args, "use_judge", False):
        print("  ! 未提供 API Key，退回离线启发式裁判")
    return heuristic_judge if getattr(args, "use_judge", False) else None


if __name__ == "__main__":
    # 自测：judge 对"同义不同措辞"应给高分，对"不同问题"应给低分
    g = {"type": "timeline", "quote": "钟楼那晚已经过去九天", "note": "应为 5 天"}
    p_same = {"type": "timeline", "quote": "已经过去九天", "issue": "时间推算有误，钟楼是 11 月 4 日"}
    p_other = {"type": "ai_flavor", "quote": "雪落了", "issue": "短句成句当金句"}
    print("同义：", heuristic_judge(p_same, g))
    print("异义：", heuristic_judge(p_other, g))
    assert heuristic_judge(p_same, g) > heuristic_judge(p_other, g)
    assert ISSUE_TYPES and norm_text(" Ａ b ") == "Ab"
    print("judge 自测通过")
