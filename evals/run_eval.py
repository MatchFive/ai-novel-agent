"""一致性审校评测 runner：跑通「评测集 → 模型 → 指标 → 报告」的完整闭环。

三种 provider（数据来源）：
  gold    用 gold 当预测 → 自检：指标必须全 1.0（验证指标实现本身没写错，这是第一步）
  noisy   在 gold 上注入漏检/误报 → 自检：指标必须按预期方向变化（验证指标是敏感的）
  api     真实调用 OpenAI 兼容模型（读提示词模板，保真）

三种 strategy（被测方案，用于做对比实验）：
  single           单次扫描（现状基线）
  self_consistency 同一条跑 N 次取交集（多数票），压误报
  verify           两阶段：扫描 → 验证器逐条裁决（先召回后精准）

示例：
  python run_eval.py --provider gold
  python run_eval.py --provider noisy --drop 0.25 --noise 0.4 --seed 7
  python run_eval.py --provider api --strategy single --model gpt-4o-mini --base-url https://api.openai.com/v1
  python run_eval.py --provider api --strategy self_consistency --repeat 3
  python run_eval.py --provider api --strategy verify
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import format_report, compute_metrics, match_issues, quote_similarity  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASET = HERE / "datasets" / "consistency_v1.jsonl"
RESULTS = HERE / "results"

# 完整版提示词文件（结构：{prompt_key: {"template": "..."}}）。
# 本仓库是"架构与评测展示层"，不含线上完整的提示词资产，该文件默认不存在 →
# 自动回退到下方与线上**结构等价**的精简模板，保证评测链路可独立跑通。
# 想评测自己的提示词：PROMPTS_JSON=/path/to/your/prompts.json python run_eval.py --provider api
PROMPTS_JSON = Path(
    os.environ.get("PROMPTS_JSON") or (HERE.parent / "backend" / "resources" / "prompts.json")
)


# ---------------------------------------------------------------------------
# 提示词装配：有完整模板时用完整模板，否则用等价精简版（评测的是同一条链路）
# ---------------------------------------------------------------------------

def load_prompt_template(prompt_key: str = "task.consistency_scan") -> str:
    if PROMPTS_JSON.exists():
        data = json.loads(PROMPTS_JSON.read_text(encoding="utf-8"))
        tpl = (data.get(prompt_key) or {}).get("template")
        if tpl:
            return tpl
    # 兜底：模板文件缺失时用与线上等价的精简版，保证评测脚本可独立运行
    return (
        "你是编辑审校。对照铁律设定、角色卡、角色经历档案、知识状态、承诺跟踪，审查正文，"
        "找出七类问题：timeline / setting / character / memory_conflict / knowledge_leak / "
        "promise_break / ai_flavor。\n"
        "【铁律设定】{{canonical_entries}}\n【角色卡】{{cast_cards}}\n"
        "【角色经历档案】{{character_memories}}\n【知识状态】{{knowledge_state}}\n"
        "【未回收承诺】{{open_promises}}\n【待审正文】{{draft_text}}\n"
        "【输出格式】严格 JSON 数组：[{\"chapter\": 0, \"quote\": \"...\", \"type\": \"...\", "
        "\"issue\": \"...\", \"suggestion\": \"...\"}]"
    )


def render_prompt(case: dict, template: str) -> str:
    ctx = dict(case["context"])
    ctx.update(case.get("context_overrides") or {})
    ctx["draft_text"] = case["draft_text"]
    out = template
    for k, v in ctx.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


# ---------------------------------------------------------------------------
# LLM 调用（只用标准库，避免给项目加新依赖）
# ---------------------------------------------------------------------------

def call_llm(prompt: str, *, base_url: str, api_key: str, model: str,
             temperature: float = 0.1, timeout: int = 120) -> tuple[str, dict]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    latency = time.time() - t0
    text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    usage = body.get("usage") or {}
    return text, {"latency": round(latency, 2),
                  "prompt_tokens": usage.get("prompt_tokens"),
                  "completion_tokens": usage.get("completion_tokens"),
                  "raw": body}


def parse_issues(text: str) -> list[dict]:
    """从模型输出里抠出 JSON 数组。模型常带 ```json 围栏或前后废话。"""
    if not text:
        return []
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if m:
        text = m.group(1)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:   # 末尾多余逗号等常见小病
            data = json.loads(re.sub(r",\s*([\]}])", r"\1", m.group(0)))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = data.get("issues") or []
    out = []
    for it in data if isinstance(data, list) else []:
        if isinstance(it, dict) and (it.get("issue") or it.get("quote")):
            out.append({"type": it.get("type") or "setting", "quote": it.get("quote") or "",
                        "issue": it.get("issue") or "", "suggestion": it.get("suggestion") or ""})
    return out


VERIFY_PROMPT = """你是复审编辑。下面有【参考资料】和一条【被提出的疑似问题】。
请判断这条问题是否**真的成立**（有依据、不是凑数、不是对合法情节的误判）。
只输出 JSON：{{"valid": true/false, "reason": "一句话"}}

【参考资料】
{context}

【待审正文】
{draft_text}

【被提出的疑似问题】
type={itype}
quote={iquote}
issue={iissue}
建议={isuggestion}
"""


def dedupe_issues(issues: list[dict], threshold: float = 0.72) -> list[dict]:
    """按 quote 相似度去重（保留先出现的）。"""
    kept: list[dict] = []
    for it in issues:
        if any(quote_similarity(it.get("quote"), k.get("quote")) >= threshold for k in kept):
            continue
        kept.append(it)
    return kept


def vote_issues(runs: list[list[dict]], keep_ratio: float = 0.5) -> list[dict]:
    """self-consistency：同一 case 多次扫描，只保留在 ≥keep_ratio 次里出现的 issue（多数票）。"""
    n = len(runs)
    if n <= 1:
        return runs[0] if runs else []
    need = max(1, int(n * keep_ratio + 0.5))
    clusters: list[dict] = []
    for run in runs:
        for it in run:
            for c in clusters:
                if quote_similarity(it.get("quote"), c["repr"].get("quote")) >= 0.7:
                    c["votes"] += 1
                    c["members"].append(it)
                    break
            else:
                clusters.append({"repr": it, "votes": 1, "members": [it]})
    return [c["repr"] for c in clusters if c["votes"] >= need]


# ---------------------------------------------------------------------------
# provider 实现
# ---------------------------------------------------------------------------

def preds_gold(cases: list[dict], args) -> dict[str, list[dict]]:
    """自检 A：gold 当预测，指标必须 = 1.0。"""
    return {c["id"]: [{"type": g["type"], "quote": g["quote"],
                       "issue": g.get("note", ""), "suggestion": ""} for g in c["gold_issues"]]
            for c in cases}


def preds_noisy(cases: list[dict], args) -> dict[str, list[dict]]:
    """自检 B：概率性地漏掉 gold、注入三类噪声，指标必须朝坏的方向动。

    注：每个 case 大多只有 1 条 gold，所以注入必须走概率，不能用 len*ratio 取整（会全被舍成 0）。
    噪声刻意分三类，分别打不同的指标：
      - hallucinated：报一段正文里根本不存在的原文 → 同时压低 precision 与（若被认领）抬高错配
      - type_swap：quote 对得上但类型判错 → 只压低 type_accuracy，不影响 recall
      - spurious：凭空多报一条 → 只压低 precision
    """
    rnd = random.Random(args.seed)
    spurious_pool = [
        {"type": "ai_flavor", "quote": "他抬起头", "issue": "疑似套话" + "（" * 0 + "）", "suggestion": ""},
        {"type": "timeline", "quote": "第二天", "issue": "疑似时间跳跃", "suggestion": ""},
        {"type": "character", "quote": "她笑了笑", "issue": "疑似腔调不符", "suggestion": ""},
    ]
    other_types = ["timeline", "setting", "character", "memory_conflict",
                   "knowledge_leak", "promise_break", "ai_flavor"]
    out: dict[str, list[dict]] = {}
    for c in cases:
        preds: list[dict] = []
        for g in c["gold_issues"]:
            if rnd.random() < args.drop:          # 漏检
                continue
            item = {"type": g["type"], "quote": g["quote"],
                    "issue": g.get("note", ""), "suggestion": ""}
            if rnd.random() < args.type_error:    # 类型判错
                options = [t for t in other_types if t != g["type"]]
                item["type"] = rnd.choice(options)
            preds.append(item)
        if rnd.random() < args.noise:             # 凭空多报
            preds.append(dict(rnd.choice(spurious_pool)))
        if rnd.random() < args.noise * 0.4:       # 幻觉原文
            preds.append({"type": "setting", "quote": "他从怀里掏出一柄铁剑抵住门框",
                          "issue": "疑似与设定冲突（原文不存在）", "suggestion": ""})
        out[c["id"]] = preds
    return out


def call_llm_retry(prompt: str, *, base_url: str, api_key: str, model: str,
                   temperature: float = 0.1, timeout: int = 120,
                   retries: int = 2, on_retry=None) -> tuple[str, dict]:
    """带指数退避的重试包装。

    为什么必须有：一次 120s 读超时会把整条 case 变成「没有预测」，而原实现把
    「没有预测」当成「模型报出 0 条」。落在干净章上就是白捡一个满分，落在 bug 章
    上就是凭空多一次漏检。网络抖动不该被算成模型能力，所以先重试。
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return call_llm(prompt, base_url=base_url, api_key=api_key, model=model,
                            temperature=temperature, timeout=timeout)
        except urllib.error.HTTPError as e:
            last = e
            # 4xx 是配置/请求问题（key 无效、模型名错、参数不支持），重试只是白等；
            # 408（超时）与 429（限流）例外，它们本质是暂时性的。
            if e.code not in (408, 429) and 400 <= e.code < 500:
                raise
            if attempt < retries:
                if on_retry:
                    on_retry(attempt + 1, e)
                time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last = e
            if attempt < retries:
                if on_retry:
                    on_retry(attempt + 1, e)
                time.sleep(2 ** attempt)      # 1s, 2s
    raise last  # type: ignore[misc]


def preds_api(cases: list[dict], args) -> tuple[dict[str, list[dict]], dict]:
    template = load_prompt_template()
    api_key = args.api_key or os.environ.get("EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("缺少 API Key：传 --api-key 或设环境变量 EVAL_API_KEY / OPENAI_API_KEY")
    out: dict[str, list[dict]] = {}
    stats = {"calls": 0, "retries": 0, "latency": [], "prompt_tokens": 0,
             "completion_tokens": 0, "errors": 0, "failed_cases": []}
    for c in cases:
        prompt = render_prompt(c, template)
        runs: list[list[dict]] = []
        repeats = args.repeat if args.strategy == "self_consistency" else 1
        for _ in range(repeats):
            try:
                text, meta = call_llm_retry(
                    prompt, base_url=args.base_url, api_key=api_key,
                    model=args.model, temperature=args.temperature,
                    retries=args.retries,
                    on_retry=lambda n, e: stats.__setitem__("retries", stats["retries"] + 1))
                stats["calls"] += 1
                stats["latency"].append(meta["latency"])
                stats["prompt_tokens"] += meta.get("prompt_tokens") or 0
                stats["completion_tokens"] += meta.get("completion_tokens") or 0
                runs.append(parse_issues(text))
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                print(f"  ! {c['id']} 调用失败（重试 {args.retries} 次后仍失败）：{e}")
                runs.append(None)          # None = 没有预测，与「报出 0 条」严格区分

        # self_consistency 只对成功的轮次投票；全部失败才算这条 case 未覆盖
        good = [r for r in runs if r is not None]
        if not good:
            stats["errors"] += 1
            stats["failed_cases"].append(c["id"])
            print(f"  {c['id']}: 未覆盖（不计入指标）")
            continue

        issues = vote_issues(good) if args.strategy == "self_consistency" else good[0]
        issues = dedupe_issues(issues)

        if args.strategy == "verify" and issues:
            kept: list[dict] = []
            for it in issues:
                ctx = dict(c["context"])
                ctx.update(c.get("context_overrides") or {})
                vp = VERIFY_PROMPT.format(
                    context="\n".join(f"【{k}】{v}" for k, v in ctx.items()),
                    draft_text=c["draft_text"], itype=it["type"], iquote=it["quote"],
                    iissue=it["issue"], isuggestion=it["suggestion"])
                try:
                    text, meta = call_llm(vp, base_url=args.base_url, api_key=api_key,
                                          model=args.verify_model or args.model, temperature=0.0)
                    stats["calls"] += 1
                    stats["latency"].append(meta["latency"])
                    stats["prompt_tokens"] += meta.get("prompt_tokens") or 0
                    stats["completion_tokens"] += meta.get("completion_tokens") or 0
                    m = re.search(r"\{.*\}", text, re.S)
                    ok = bool(m and json.loads(m.group(0)).get("valid"))
                except Exception:      # noqa: BLE001  验证器失败时保守保留（宁可多报不漏报）
                    ok = True
                if ok:
                    kept.append(it)
            issues = kept

        out[c["id"]] = issues
        print(f"  {c['id']}: 报出 {len(issues)} 条")

    if stats["latency"]:
        stats["latency_p50"] = round(statistics.median(stats["latency"]), 2)
        stats["latency_max"] = round(max(stats["latency"]), 2)
    return out, stats


# ---------------------------------------------------------------------------

def load_cases() -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_metrics(m: dict, expect: dict) -> None:
    """按预期值逐项核对指标（用于确定性自检：gold 必须全 1.0，noisy 必须等于既定基线）。

    自检口径是确定性的（噪声注入带固定 seed），所以用精确相等——任何漂移都该让门禁响。
    真实模型输出有抖动，不能用这个，见 assert_thresholds。
    """
    bad = {k: (m.get(k), v) for k, v in expect.items() if m.get(k) != v}
    if bad:
        detail = "\n".join(f"  {k}: 实际 {a!r} ≠ 预期 {e!r}" for k, (a, e) in bad.items())
        raise SystemExit(f"指标与预期不符：\n{detail}")


def assert_thresholds(m: dict, minimums: dict | None = None,
                      maximums: dict | None = None) -> str:
    """按上下限核对指标，用于真实模型门禁。

    为什么不复用 assert_metrics：真实模型不是确定性的，同一条 prompt 两次跑
    recall 会差几个点，精确相等必然误红。这里只守「不能更差」的地板。

    maximums 存在的理由：clean_fp_rate（干净章平均误报条数）越低越好——只设下限
    会漏掉「把所有章都报一遍」这种最典型的坏模型。
    """
    minimums, maximums = minimums or {}, maximums or {}
    bad: list[str] = []
    remarks: list[str] = []
    for k, floor in minimums.items():
        actual = m.get(k)
        if actual is None:
            bad.append(f"  {k}: 指标不存在（预期下限 {floor}）")
        elif actual < floor:
            bad.append(f"  {k}: 实际 {actual} < 下限 {floor}")
        else:
            remarks.append(f"{k}={actual} ≥ {floor}")
    for k, ceil in maximums.items():
        actual = m.get(k)
        if actual is None:
            bad.append(f"  {k}: 指标不存在（预期上限 {ceil}）")
        elif actual > ceil:
            bad.append(f"  {k}: 实际 {actual} > 上限 {ceil}")
        else:
            remarks.append(f"{k}={actual} ≤ {ceil}")
    if bad:
        raise SystemExit("指标未达门禁要求：\n" + "\n".join(bad))
    return "，".join(remarks)


def main() -> None:
    ap = argparse.ArgumentParser(description="一致性审校评测 runner")
    ap.add_argument("--provider", choices=["gold", "noisy", "api"], default="gold")
    ap.add_argument("--strategy", choices=["single", "self_consistency", "verify"], default="single")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--verify-model", default=None)
    ap.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2,
                    help="单次调用失败后的重试次数（指数退避 1s/2s）。网络抖动不该算成模型能力")
    ap.add_argument("--drop", type=float, default=0.25, help="noisy 模式：每条 gold 被漏掉的概率")
    ap.add_argument("--noise", type=float, default=0.35, help="noisy 模式：每个 case 注入多余问题的概率")
    ap.add_argument("--type-error", type=float, default=0.2, help="noisy 模式：命中的问题被判错类型的概率")
    ap.add_argument("--use-judge", action="store_true",
                    help="启用 L2 裁判兜底匹配（有 Key 走 LLM 裁判，无 Key 走离线启发式裁判）")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default=None, help="本次运行的标签，用于对比表")
    ap.add_argument("--expect-json", default=None,
                    help='门禁用（确定性自检）：预期指标 JSON，如 \'{"recall":1.0,"precision":1.0}\'，不符则非零退出')
    ap.add_argument("--min-json", default=None,
                    help='门禁用（真实模型）：指标下限 JSON，如 \'{"recall":0.6,"precision":0.45}\'，'
                         '任一低于下限则非零退出；指标名写错/缺失也判失败，避免静默放行')
    ap.add_argument("--max-json", default=None,
                    help='门禁用（真实模型）：指标上限 JSON，如 \'{"clean_fp_rate":0.8}\'，'
                         '任一超出上限则非零退出（越小越好的指标用这个，如误报率）')
    args = ap.parse_args()

    cases = load_cases()
    print(f"载入 {len(cases)} 条 case（{DATASET.name}）")

    from judge import get_judge
    judge = get_judge(args)

    stats = {}
    if args.provider == "gold":
        preds = preds_gold(cases, args)
    elif args.provider == "noisy":
        preds = preds_noisy(cases, args)
    else:
        print(f"真实调用：model={args.model} strategy={args.strategy}")
        preds, stats = preds_api(cases, args)

    m = compute_metrics(cases, preds, judge=judge,
                        skip_ids=set(stats.get("failed_cases") or [])) if judge \
        else compute_metrics(cases, preds, skip_ids=set(stats.get("failed_cases") or []))
    tag = args.tag or f"{args.provider}-{args.strategy}" + ("-judge" if judge else "")
    run_dir = RESULTS / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for cid, issues in preds.items():
            f.write(json.dumps({"id": cid, "issues": issues}, ensure_ascii=False) + "\n")
    (run_dir / "metrics.json").write_text(
        json.dumps({"config": vars(args), "stats": stats, "metrics": {k: v for k, v in m.items() if k != "details"}},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {"数据集": DATASET.name, "provider": args.provider, "strategy": args.strategy,
            "model": args.model if args.provider == "api" else "-"}
    if stats:
        meta["调用次数"] = stats.get("calls")
        if stats.get("retries"):
            meta["重试次数"] = stats["retries"]
        meta["延迟 p50 / max (s)"] = f"{stats.get('latency_p50')} / {stats.get('latency_max')}"
        meta["token 合计（in/out）"] = f"{stats.get('prompt_tokens')} / {stats.get('completion_tokens')}"
        meta["失败调用"] = stats.get("errors")
        if stats.get("failed_cases"):
            meta["未覆盖 case"] = ", ".join(stats["failed_cases"])
    report = format_report(m, title=f"一致性审校评测报告 · {tag}", meta=meta)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    # 汇总表：每次运行追加一行，便于横向对比不同 prompt / 策略
    summary = RESULTS / "summary.csv"
    header = ("tag,provider,strategy,model,recall,must_recall,weighted_recall,precision,f1,"
              "type_accuracy,clean_fp_rate,gold,pred,calls,latency_p50,coverage\n")
    if not summary.exists():
        summary.write_text(header, encoding="utf-8")
    else:
        # 旧表没有 coverage 列：补 1.0（这些运行确实没有失败 case），否则列会错位
        lines = [ln for ln in summary.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines and "coverage" not in lines[0]:
            migrated = [lines[0] + ",coverage"] + [ln + ",1.0" for ln in lines[1:]]
            summary.write_text("\n".join(migrated) + "\n", encoding="utf-8")
    with summary.open("a", encoding="utf-8") as f:
        f.write(f"{tag},{args.provider},{args.strategy},{meta['model']},{m['recall']},{m['must_detect_recall']},"
                f"{m['weighted_recall']},{m['precision']},{m['f1']},{m['type_accuracy']},{m['clean_fp_rate']},"
                f"{m['gold_total']},{m['pred_total']},{stats.get('calls', 0)},{stats.get('latency_p50', '')},"
                f"{m['coverage']}\n")

    print()
    print(report)
    print(f"报告已写入：{run_dir / 'report.md'}")
    print(f"对比表：{summary}")

    if args.expect_json:
        assert_metrics(m, json.loads(args.expect_json))
        print("指标符合预期 ✓")

    if args.min_json or args.max_json:
        detail = assert_thresholds(
            m,
            json.loads(args.min_json) if args.min_json else None,
            json.loads(args.max_json) if args.max_json else None,
        )
        print(f"指标达到门禁要求 ✓（{detail}）")


if __name__ == "__main__":
    main()
