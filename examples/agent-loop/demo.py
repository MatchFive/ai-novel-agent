"""演示：交付物门禁派生 + 审批闸门。

运行：`python demo.py`
"""
from __future__ import annotations

import asyncio

from approval_gate import (
    ApprovalGate,
    ApprovalRequest,
    allowed_tools,
    is_tool_allowed,
    resolve_deliverables,
)


def demo_gate() -> None:
    print("=" * 78)
    print("一、交付物 → 工具门禁（并集 + 依赖闭包，fail-closed）")
    print("=" * 78)

    for selected in (["world"], ["outline"], ["draft"], ["draft", "characters"]):
        print(f"\n用户选择：{selected}")
        print(f"  含依赖闭包后实际授权：{sorted(resolve_deliverables(selected))}")
        print(f"  可用工具：{sorted(allowed_tools(selected))}")

    print("\n" + "-" * 78)
    print("关键对比：用户说「只写第 3~5 章」→ deliverables = ['draft']")
    print("-" * 78)
    for tool in ("write_plot", "split_chapters", "build_outline", "build_world", "build_characters"):
        verdict = "允许" if is_tool_allowed(tool, ["draft"]) else "拒绝（fail-closed）"
        print(f"  {tool:20s} {verdict}")

    print("\n  draft 带上了 chapters / outline（没细纲写不了正文），")
    print("  但**没有**带上 world / characters——「只写正文」不会顺势扩建成重建世界观。")
    print("  注意这是白名单派生，不是维护一份禁止清单：新工具默认不可用。")


async def demo_approval() -> None:
    print("\n" + "=" * 78)
    print("二、审批闸门：Agent 停下来等人，人回来了接着走")
    print("=" * 78)

    gate = ApprovalGate(timeout=5.0)
    req = ApprovalRequest("req-1", kind="plan",
                          payload={"volumes": 3, "plots": 12, "chapters": 40})

    async def human() -> None:
        await asyncio.sleep(0.3)          # 模拟人在思考
        ok = gate.resolve("req-1", approved=True,
                          payload={"note": "第二卷的反派提前登场"})
        print(f"  [人工] 提交裁决，闸门已唤醒：{ok}")

    print("  [Agent] 规划完成，请求人工确认（挂起等待…）")
    results = await asyncio.gather(gate.wait(req), human())
    result = results[0]
    print(f"  [Agent] 收到裁决：{result.status}  payload={result.payload}")
    print(f"  [Agent] run 状态回到 {gate.state.value}，继续执行")

    again = gate.resolve("req-1", approved=True)
    print(f"  [人工] 重复提交同一请求 → {again}（幂等：不报错、不重复唤醒）")


async def demo_timeout() -> None:
    print("\n" + "=" * 78)
    print("三、无人响应 → paused（任务不失败，现场保留，可恢复）")
    print("=" * 78)

    gate = ApprovalGate(timeout=0.3)
    req = ApprovalRequest("req-2", kind="choice",
                          payload={"options": ["金手指A", "金手指B", "金手指C"]})
    print("  [Agent] 抛出 3 套候选方案，请求用户选择（0.3s 后超时）")
    result = await gate.wait(req)
    print(f"  → status={result.status}  reason={result.payload.get('reason')}")
    print(f"  run 状态：{gate.state.value}——不是 failed，用户回来可继续推进")
    late = gate.resolve("req-2", approved=True)
    print(f"  超时后再提交裁决 → {late}（请求已清理，安全返回 False）")


if __name__ == "__main__":
    demo_gate()
    asyncio.run(demo_approval())
    asyncio.run(demo_timeout())
    print("\n" + "=" * 78)
    print("要点")
    print("=" * 78)
    print("1. 门禁用白名单 + 依赖闭包确定性派生，fail-closed；不靠模型自觉，也不靠禁止清单。")
    print("2. 人工介入要能挂起、能超时、能恢复，且裁决幂等（接口重试是常态）。")
    print("3. 两道门分工不同：规划确认审结构，选项确认让人做选择——别只给一个文本框。")
