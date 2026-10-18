"""把一致性审校评测接到 LangSmith：可视化 trace + 横向实验对比 + 团队可复现。

本地 CSV 报告解决"我自己知道分数"，LangSmith 解决三件本地做不到的事：
  1. **可追溯**：每次 run 的输入/输出/prompt 版本/延迟全留痕，能点开看模型当时看到了什么；
  2. **可对比**：experiment 之间直接并排看，改 prompt 前后逐条 diff；
  3. **可协作**：把 dataset 与实验分享出去，别人能一键复现——面试时这就是"工程规范"的证据。

准备：
  pip install langsmith
  export LANGSMITH_API_KEY=lsv2_...
  export LANGSMITH_TRACING=true          # （可选）开启 trace
  export EVAL_API_KEY=...                # 被测模型 Key

用法：
  python langsmith_eval.py --upload-only                      # 只上传数据集
  python langsmith_eval.py --model gpt-4o-mini               # 上传 + 跑一轮实验
  python langsmith_eval.py --strategy self_consistency --repeat 3
  python langsmith_eval.py --upload-only --force             # 删掉旧数据集重建
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import ISSUE_TYPES, match_issues, quote_similarity  # noqa: E402
from judge import heuristic_judge                                # noqa: E402
from run_eval import (DATASET, dedupe_issues, load_cases, load_prompt_template,  # noqa: E402
                      parse_issues, render_prompt, vote_issues, call_llm)

DATASET_NAME = "novel-consistency-v1"


# ---------------------------------------------------------------------------
# 数据集上传
# ---------------------------------------------------------------------------

def upload_dataset(client, force: bool = False):
    if force:
        for ds in client.list_datasets(dataset_name=DATASET_NAME):
            client.delete_dataset(dataset_id=ds.id)
            print(f"已删除旧数据集 {DATASET_NAME}")

    try:
        ds = next(iter(client.list_datasets(dataset_name=DATASET_NAME)))
        print(f"复用已有数据集：{DATASET_NAME}（{ds.id}）")
    except StopIteration:
        ds = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="AI 小说一致性审校评测集 v1.1：15 条人工种 bug 章（23 处 gold）+ 3 条干净章 + 2 条诱饵章，"
                        "覆盖 timeline / setting / character / memory_conflict / knowledge_leak / "
                        "promise_break / ai_flavor 七类问题。",
        )
        cases = load_cases()
        template = load_prompt_template()
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{
                "case_id": c["id"],
                "category": c["category"],
                "difficulty": c["difficulty"],
                "draft_text": c["draft_text"],
                "context": c["context"],
                "prompt": render_prompt(c, template),   # 线上真模板渲染后的完整 prompt
            } for c in cases],
            outputs=[{
                "gold_issues": c["gold_issues"],
                "category": c["category"],
                "type_focus": c.get("type_focus"),
            } for c in cases],
            metadata=[{"id": c["id"], "category": c["category"],
                       "difficulty": c["difficulty"]} for c in cases],
        )
        print(f"已上传 {len(cases)} 条 example 到 {DATASET_NAME}")
    return ds


# ---------------------------------------------------------------------------
# 被测目标（target）：LangSmith 每个 example 调一次
# ---------------------------------------------------------------------------

def build_target(args):
    api_key = args.api_key or os.environ.get("EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("缺少 API Key：--api-key 或 EVAL_API_KEY / OPENAI_API_KEY")

    def target(inputs: dict) -> dict:
        runs: list[list[dict]] = []
        repeats = args.repeat if args.strategy == "self_consistency" else 1
        for _ in range(repeats):
            text, _meta = call_llm(inputs["prompt"], base_url=args.base_url, api_key=api_key,
                                   model=args.model, temperature=args.temperature)
            runs.append(parse_issues(text))
        issues = vote_issues(runs) if args.strategy == "self_consistency" else (runs[0] if runs else [])
        issues = dedupe_issues(issues)
        return {"issues": issues, "issue_count": len(issues)}

    return target


# ---------------------------------------------------------------------------
# 评估器（evaluator）：把 metrics.py 的算法原样搬过来，保证两套口径一致
# ---------------------------------------------------------------------------

def consistency_evaluators():
    judge = heuristic_judge   # 需要更准时改这里换成 build_llm_judge(...)

    def _r(run, example):
        preds = (run.outputs or {}).get("issues") or []
        golds = (example.outputs or {}).get("gold_issues") or []
        category = (example.outputs or {}).get("category") or "gold_bug"
        r = match_issues(preds, golds, judge=judge)

        if golds:
            recall = len(r["pairs"]) / len(golds)
        else:   # 干净章/诱饵章：没有 gold，用"零误报"记满分
            recall = 1.0 if not r["unmatched_pred"] else 0.0
        precision = len(r["pairs"]) / len(preds) if preds else 1.0
        type_ok = (sum(1 for p in r["pairs"] if p["type_ok"]) / len(r["pairs"])) if r["pairs"] else 1.0

        return [
            {"key": "recall", "score": round(recall, 4)},
            {"key": "precision", "score": round(precision, 4)},
            {"key": "f1", "score": round(2 * recall * precision / (recall + precision), 4)
             if (recall + precision) else 0.0},
            {"key": "type_accuracy", "score": round(type_ok, 4)},
            {"key": "false_positives", "score": len(r["unmatched_pred"])},
            {"key": "is_clean_case_fp", "score": 1 if (category in ("clean", "decoy")
                                                      and r["unmatched_pred"]) else 0},
        ]

    return [_r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--strategy", choices=["single", "self_consistency"], default="single")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--experiment-prefix", default="consistency-scan")
    ap.add_argument("--upload-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="删除并重建数据集")
    args = ap.parse_args()

    try:
        from langsmith import Client, evaluate
    except ImportError:
        raise SystemExit("未安装 langsmith：pip install langsmith")

    if not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit("缺少 LANGSMITH_API_KEY（在 https://smith.langchain.com 获取）")

    client = Client()
    ds = upload_dataset(client, force=args.force)
    if args.upload_only:
        print("仅上传模式，结束。")
        return

    print(f"开始实验：model={args.model} strategy={args.strategy}")
    results = evaluate(
        build_target(args),
        data=DATASET_NAME,
        evaluators=consistency_evaluators(),
        experiment_prefix=f"{args.experiment_prefix}-{args.strategy}",
        metadata={"model": args.model, "strategy": args.strategy,
                  "temperature": args.temperature, "dataset": DATASET.name},
        max_concurrency=2,
    )
    print("\n实验完成。到 LangSmith 项目页对比 experiment，或：")
    print(f"  from langsmith import Client; Client().get_experiment_results('{results.experiment_name}')")


if __name__ == "__main__":
    main()
