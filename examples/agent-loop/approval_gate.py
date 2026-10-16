"""人工审批闸门 + fail-closed 工具门禁（骨架实现）。

从生产代码简化而来：去掉数据库持久化与 Web 层，只保留三样真正重要的东西——
**并发控制、门禁判定、状态机**。

为什么要这两样东西：

1. **Agent 停下来等人，不能死等。** 自主创作的 Agent 会一口气把整本书规划完再交付，
   中途需要人审。但人可能半小时后才回来——所以等待必须有超时，超时后转 `paused`
   并保留现场，人回来能接着走，而不是任务卡死或整轮重跑。
2. **"能做什么"不能交给模型自觉。** 用户说「只写第 3~5 章」，Agent 却有权限重建世界观——
   早晚会顺手把世界观改掉。所以用**封闭词表 + 依赖闭包**确定性地派生工具集，
   且 **fail-closed**（未列出的一律拒绝），而不是维护一份"禁止清单"。
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# 运行状态
# ---------------------------------------------------------------------------


class RunState(str, Enum):
    RUNNING = "running"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    AWAITING_USER_CHOICE = "awaiting_user_choice"
    PAUSED = "paused"          # 审批超时，现场保留，可恢复
    DONE = "done"


# ---------------------------------------------------------------------------
# 交付物 → 工具集：封闭词表 + 依赖闭包
# ---------------------------------------------------------------------------

# 每个交付物允许使用的工具
DELIVERABLE_TOOLS: dict[str, set[str]] = {
    "world":      {"build_world", "write_world_entry"},
    "characters": {"build_characters", "write_character"},
    "outline":    {"build_outline", "write_plot"},
    "chapters":   {"split_chapters", "write_chapter_plan"},
    "draft":      {"write_plot"},          # 写正文经由 write_plot
    "edit":       {"edit_by_command"},
}

# 依赖闭包：声明「产出 A 必须先有 B」
#
# 关键在 draft：它依赖 chapters，**但不依赖 world / characters**。
# 否则「只写正文」会被顺势扩建成重建世界观——这正是旧版「四档枚举」的毛病：
# 「只写第 3~5 章」只能选 full，等于默许动所有东西。
DEPENDENCIES: dict[str, set[str]] = {
    "chapters": {"outline"},
    "draft":    {"chapters"},
}

# 任何交付物下都可用的工具：只读、提问、审批、收尾
ALWAYS_AVAILABLE = frozenset({
    "search", "read_context",
    "ask_user", "request_plan_approval",
    "finish",
})


def resolve_deliverables(selected: Iterable[str]) -> frozenset[str]:
    """并集 + 依赖闭包。词表外的选择直接忽略（fail-closed）。"""
    out: set[str] = set()
    for item in selected:
        if item not in DELIVERABLE_TOOLS:
            continue
        out.add(item)
        out |= DEPENDENCIES.get(item, set())
    return frozenset(out)


def allowed_tools(deliverables: Iterable[str]) -> frozenset[str]:
    """由交付物确定性地派生可用工具集。"""
    tools: set[str] = set(ALWAYS_AVAILABLE)
    for item in resolve_deliverables(deliverables):
        tools |= DELIVERABLE_TOOLS[item]
    return frozenset(tools)


def is_tool_allowed(tool: str, deliverables: Iterable[str]) -> bool:
    """fail-closed：不在白名单里就是不可用，没有例外。"""
    return tool in allowed_tools(deliverables)


# ---------------------------------------------------------------------------
# 审批闸门
# ---------------------------------------------------------------------------


@dataclass
class ApprovalRequest:
    request_id: str
    kind: str                 # plan 规划确认 / choice 选项确认
    payload: dict = field(default_factory=dict)


@dataclass
class ApprovalResult:
    status: str               # approved / rejected / paused
    payload: dict = field(default_factory=dict)


class ApprovalGate:
    """人工审批闸门。

    线上两道门（规划确认 / 选项确认）复用同一套机制：`asyncio.Event` + 900s 超时。
    分开两道门是因为它们问的东西不同——
    规划确认审的是结构化的卷/情节/章节，选项确认是把「在多个方案里挑一个」交给人，
    只给自由文本输入是没用的（真实事故：Agent 抛出 5 套候选金手指，前端没有可点的东西，
    用户只能自己打字，而任务已经标记完成）。
    """

    def __init__(self, timeout: float = 900.0) -> None:
        self._timeout = timeout
        self._events: dict[str, asyncio.Event] = {}
        self._results: dict[str, ApprovalResult] = {}
        self.state: RunState = RunState.RUNNING

    async def wait(self, req: ApprovalRequest) -> ApprovalResult:
        """挂起等待人工裁决；超时转 paused（现场保留，可恢复）。"""
        ev = asyncio.Event()
        self._events[req.request_id] = ev
        self.state = (
            RunState.AWAITING_PLAN_APPROVAL if req.kind == "plan"
            else RunState.AWAITING_USER_CHOICE
        )
        try:
            await asyncio.wait_for(ev.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            self._events.pop(req.request_id, None)
            self.state = RunState.PAUSED
            return ApprovalResult(status="paused", payload={"reason": "approval_timeout"})
        self.state = RunState.RUNNING
        return self._results.pop(req.request_id)

    def resolve(self, request_id: str, *, approved: bool, payload: dict | None = None) -> bool:
        """人工提交裁决。

        幂等：只有仍在等待的请求会被唤醒。重复提交、或对已超时清理的请求提交，
        都返回 False 而不报错——接口重试是常态，不能因此把任务搞乱。
        """
        ev = self._events.pop(request_id, None)
        if ev is None:
            return False
        self._results[request_id] = ApprovalResult(
            status="approved" if approved else "rejected",
            payload=payload or {},
        )
        ev.set()
        return True

    @property
    def pending(self) -> list[str]:
        return list(self._events)
