"""构建一致性审校评测集 v1.1（20 条）。

v1.1（run #5 复盘后修订）：ai_flavor 是「密集型」问题——同一章里往往有多处独立套路，
v1 每条 case 只标 1 处 gold，导致模型多报的每一处真问题都被记成误报（precision 被口径
压成下界）。v1.1 为密集 case 补标全部可独立取证的套路实例（severity=minor、不计必检出，
避免改变必检出分母），并给 tc14 主 gold 补 acceptable_quotes 多锚点。

产出：
  datasets/context_base.json      —— 共享世界上下文（与 check_service._scan_context 的 5 个变量同名）
  datasets/consistency_v1.jsonl   —— 20 条 case（15 条人工种 bug + 3 条干净章 + 2 条诱饵章）

用法：
  python datasets/build_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import longest_common_run, norm_text  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 共享世界上下文：字段名严格对齐 CheckService._scan_context() 的返回值，
# 这样 runner 可以直接把它们塞进 task.consistency_scan 模板的 5 个变量。
# 世界观：《霜降之城》——都市奇幻悬疑。所有 case 共用同一套设定与人物，
# 目标是"逐章审校"，考验模型跨文档对照的能力。
# ---------------------------------------------------------------------------
SHARED = {
    "canonical_entries": "\n".join([
        "【霜纹】被霜蚀者皮肤上会浮现蓝色纹路，触碰活物会瞬间结霜（近 -30℃）。"
        "霜纹只能作用于木、石、水与空气；接触铁器会逆流剧痛，霜蚀者在持铁状态下无法施术。",
        "【铜】铜器不引霜纹逆流，可作贴身器物。",
        "【回溯】霜蚀者每七天必须在无光环境中静置一小时，否则霜纹失控。",
        "【时间锚】故事始于 11 月 3 日；第 9 章为 11 月 4 日（钟楼）；霜降为 11 月 7 日；本章为 11 月 9 日。",
    ]),
    "cast_cards": "\n".join([
        "【沈砚】(主角) 男 32，气象观测员，霜蚀者，双手有霜纹。性格克制、寡言，惯用短句与冷嘲，"
        "随身带一块铜制怀表。",
        "【陆知雨】女 26，晚报记者，普通人。父亲三年前失踪，行动力强，说话直。不具备任何霜纹感知能力。",
        "【老周】男 60，修表匠，第一批霜蚀者，知道「回溯」以外压制霜纹的方法，从不饮酒。",
    ]),
    "character_memories": "\n".join([
        "【沈砚】第 7 章在冷库用霜纹冻住追兵的货车门（已动用过霜纹）；"
        "第 9 章（11 月 4 日）在钟楼被铁链灼伤，卧床三天至 11 月 6 日。",
        "【陆知雨】第 11 章在冷库外隔着门缝看见沈砚手上的蓝纹，但未确认那是霜纹。",
        "【老周】第 5 章对沈砚说过「铜能替铁」；从不饮酒。",
    ]),
    "knowledge_state": "\n".join([
        "【霜蚀者存在】知晓者：沈砚、老周（陆知雨不知晓，仅怀疑沈砚「不普通」）",
        "【老周是初代霜蚀者】知晓者：老周",
        "【回溯的规则】知晓者：沈砚、老周",
        "【怀表的来历】知晓者：无人",
    ]),
    "open_promises": "\n".join([
        "（伏笔）陆知雨父亲的旧怀表——第 3 章埋，尚未揭示来源",
        "（承诺）老周承诺：霜降那天（11 月 7 日）告诉沈砚「霜纹的来历」——做出者：老周",
        "（承诺）陆知雨答应报社主编：在霜降前交出一篇能上头条的稿子——做出者：陆知雨",
    ]),
}

E, M, H = "easy", "medium", "hard"

CASES: list[dict] = [
    # ---------------- timeline ----------------
    {
        "id": "tc01", "category": "gold_bug", "type_focus": "timeline", "difficulty": E, "chapter": 12,
        "draft_text": (
            "傍晚六点，天色已经黑透。沈砚从怀表上抬起眼，往东街走。\n"
            "巷口的雪被踩得发硬。正午的阳光把霜城的屋顶照得发白，屋檐下的冰棱一根根亮起来。\n"
            "他把表塞回口袋，继续走。"
        ),
        "gold_issues": [{
            "type": "timeline", "quote": "正午的阳光把霜城的屋顶照得发白",
            "severity": "critical", "must_detect": True,
            "note": "同章内时间自相矛盾：前文「傍晚六点，天色已经黑透」，后文却是正午的阳光。",
        }],
    },
    {
        "id": "tc02", "category": "gold_bug", "type_focus": "timeline", "difficulty": M, "chapter": 12,
        "draft_text": (
            "11 月 9 日，霜降过去两天。\n"
            "钟楼那晚已经过去九天，沈砚背上的焦痕早就褪成一道浅白。他数着日子，忽然想起一件事——"
            "霜降那天，修表铺的门一直关着。老周失约了。"
        ),
        "gold_issues": [{
            "type": "timeline", "quote": "钟楼那晚已经过去九天",
            "severity": "major", "must_detect": True,
            "note": "第 9 章＝11 月 4 日，本章＝11 月 9 日，实为 5 天；「九天」需与铁律时间锚对照才能发现。"
                    "亦可判为 memory_conflict（多标签容错）。",
            "acceptable_types": ["timeline", "memory_conflict"],
        }],
    },
    # ---------------- setting ----------------
    {
        "id": "tc03", "category": "gold_bug", "type_focus": "setting", "difficulty": E, "chapter": 12,
        "draft_text": (
            "沈砚一把攥住生锈的铁门闩，掌心的霜纹骤然亮起，整扇铁门在一声脆响里冻裂开来。\n"
            "他甩了甩手，像什么都没发生。"
        ),
        "gold_issues": [{
            "type": "setting", "quote": "攥住生锈的铁门闩，掌心的霜纹骤然亮起",
            "severity": "critical", "must_detect": True,
            "note": "违反铁律：铁器引霜纹逆流剧痛，持铁状态下无法施术，更不可能冻裂铁门。",
        }],
    },
    {
        "id": "tc04", "category": "gold_bug", "type_focus": "setting", "difficulty": M, "chapter": 12,
        "draft_text": (
            "老周把茶推过去，「你今天该回溯了。」\n"
            "沈砚摇头，「三天前刚做过。」\n"
            "老周点点头，「那就没事。三天一次，只要你守时，霜纹就压得住。」"
        ),
        "gold_issues": [{
            "type": "setting", "quote": "三天一次，只要你守时，霜纹就压得住",
            "severity": "critical", "must_detect": True,
            "note": "铁律明确「每七天一次」，此处被改成三天，属于设定被篡改（老周作为知情者不该说错）。",
        }],
    },
    # ---------------- character ----------------
    {
        "id": "tc05", "category": "gold_bug", "type_focus": "character", "difficulty": E, "chapter": 12,
        "draft_text": (
            "沈砚哈哈大笑，一巴掌拍在陆知雨肩上，一连讲了三个带颜色的笑话，逗得整条巷子的人都回头。\n"
            "他向来最爱热闹，说话像连珠炮，一句接一句不带停的。"
        ),
        "gold_issues": [{
            "type": "character", "quote": "沈砚哈哈大笑，一巴掌拍在陆知雨肩上",
            "severity": "major", "must_detect": True,
            "note": "与角色卡「克制、寡言、短句冷嘲」正相反，人物腔调崩坏。",
        }, {
            "type": "ai_flavor", "quote": "他向来最爱热闹，说话像连珠炮，一句接一句不带停的。",
            "severity": "minor", "must_detect": False,
            "note": "v1.1 补标：段末总结性人设复述（「向来最爱热闹」）与「连珠炮」比喻同构并置，"
                    "是独立的 ai_flavor 实例（run #5 实测模型报出、v1 口径下被记成误报）。",
        }],
    },
    {
        "id": "tc06", "category": "gold_bug", "type_focus": "character", "difficulty": M, "chapter": 12,
        "draft_text": (
            "陆知雨闭上眼，霜纹的感应顺着地砖铺开——三公里外，两个霜蚀者正在靠近。\n"
            "她睁开眼，「来了两个，走北边那条巷子。」"
        ),
        "gold_issues": [{
            "type": "character", "quote": "霜纹的感应顺着地砖铺开——三公里外",
            "severity": "critical", "must_detect": True,
            "note": "陆知雨是普通人，角色卡明确她不具备霜纹感知能力。",
        }],
    },
    # ---------------- memory_conflict ----------------
    {
        "id": "tc07", "category": "gold_bug", "type_focus": "memory_conflict", "difficulty": E, "chapter": 12,
        "draft_text": (
            "沈砚第一次动用霜纹，是在这个下雨的夜里。\n"
            "他从前不敢，也从未试过。掌心的蓝纹一点点浮出来，他盯着看，像在看别人的手。"
        ),
        "gold_issues": [{
            "type": "memory_conflict", "quote": "沈砚第一次动用霜纹",
            "severity": "major", "must_detect": True,
            "note": "角色经历档案：第 7 章已在冷库用霜纹冻住货车门。",
        }],
    },
    {
        "id": "tc08", "category": "gold_bug", "type_focus": "memory_conflict", "difficulty": M, "chapter": 12,
        "draft_text": (
            "老周给自己倒了半杯白酒，一饮而尽。\n"
            "「修表这行，四十年前就这样了。」这是他多年的习惯，每天收摊后来上半杯，从没断过。"
        ),
        "gold_issues": [{
            "type": "memory_conflict", "quote": "半杯白酒，一饮而尽",
            "severity": "major", "must_detect": True,
            "note": "角色档案与角色卡均写明老周从不饮酒。",
        }],
    },
    # ---------------- knowledge_leak ----------------
    {
        "id": "tc09", "category": "gold_bug", "type_focus": "knowledge_leak", "difficulty": E, "chapter": 12,
        "draft_text": (
            "「你手上的霜纹，比上周又重了一圈。」陆知雨说。\n"
            "沈砚一僵，把袖子往下拉了拉。"
        ),
        "gold_issues": [{
            "type": "knowledge_leak", "quote": "你手上的霜纹，比上周又重了一圈",
            "severity": "critical", "must_detect": True,
            "note": "知识状态：陆知雨不知晓「霜蚀者/霜纹」概念，不可能说出「霜纹」这个词并做对比。",
        }],
    },
    {
        "id": "tc10", "category": "gold_bug", "type_focus": "knowledge_leak", "difficulty": H, "chapter": 12,
        "draft_text": (
            "陆知雨把名片按在柜台上，「初代霜蚀者都在哪？你是最早的那一批，这些人你应该见过全部。」\n"
            "老周手一抖，镊子掉在台面上，弹了两下。"
        ),
        "gold_issues": [{
            "type": "knowledge_leak", "quote": "你是最早的那一批",
            "severity": "major", "must_detect": True,
            "note": "知识状态：「老周是初代霜蚀者」只有老周自己知晓；"
                    "陆知雨既不知霜蚀者存在，更不该知道老周的身份——需两跳推理才能判定。",
        }],
    },
    # ---------------- promise_break ----------------
    {
        "id": "tc11", "category": "gold_bug", "type_focus": "promise_break", "difficulty": M, "chapter": 12,
        "draft_text": (
            "「你想知道霜纹的来历吧。」老周说，「我早就告诉过你了，在霜降那天。」\n"
            "沈砚点头，「嗯，你说过。」\n"
            "两个人沉默下来，茶还冒着热气。"
        ),
        "gold_issues": [{
            "type": "promise_break", "quote": "我早就告诉过你了，在霜降那天",
            "severity": "major", "must_detect": True,
            "note": "未回收承诺里该承诺仍挂着，正文却宣称已兑现（虚假回收），会导致后续剧情空洞。",
        }],
    },
    {
        "id": "tc12", "category": "gold_bug", "type_focus": "promise_break", "difficulty": H, "chapter": 12,
        "draft_text": (
            "老周从抽屉里摸出一枚铜片，塞进沈砚手里，「拿着。以后遇到铁器，用这个。」\n"
            "沈砚愣住，「铜……能替铁？我从来没听说过。」"
        ),
        "gold_issues": [{
            "type": "promise_break", "quote": "铜……能替铁？我从来没听说过",
            "severity": "minor", "must_detect": False,
            "note": "第 5 章老周已对沈砚讲过「铜能替铁」，此处主角当新知识接受＝旧信息被抛弃。"
                    "主类型可判 memory_conflict，属于类型边界案例（不计入必检出，只用于观察判定宽严）。",
            "acceptable_types": ["promise_break", "memory_conflict"],
        }],
    },
    # ---------------- ai_flavor ----------------
    {
        "id": "tc13", "category": "gold_bug", "type_focus": "ai_flavor", "difficulty": E, "chapter": 12,
        "draft_text": (
            "他不是害怕，而是终于明白了。霜城的夜不是冷，而是一种漫长的告别。\n"
            "他站在那里，像一柄终于入鞘的刀。风雪落下来的时候，一切都安静了。"
        ),
        "gold_issues": [{
            "type": "ai_flavor", "quote": "他不是害怕，而是终于明白了",
            "severity": "major", "must_detect": True,
            "note": "否定对偶「不是X，而是Y」＋比喻收尾升华，是通报里点名的 AI 套路。",
        }, {
            "type": "ai_flavor", "quote": "霜城的夜不是冷，而是一种漫长的告别",
            "severity": "minor", "must_detect": False,
            "note": "v1.1 补标：同一否定对偶在同段复现（第 2 处）。",
        }, {
            "type": "ai_flavor", "quote": "一切都安静了",
            "severity": "minor", "must_detect": False,
            "note": "v1.1 补标：短句独立成句收束段落，做总结升华（run #5 实测模型报出、v1 口径下被记成误报）。",
        }],
    },
    {
        "id": "tc14", "category": "gold_bug", "type_focus": "ai_flavor", "difficulty": M, "chapter": 12,
        "draft_text": (
            "他缓缓地深吸一口气，指节微微泛白，喉结轻轻滚动了一下，眼底深处的情绪几不可察地翻涌着。\n"
            "他缓缓地抬起头，缓缓地看向窗外，嘴角勾起一抹若有若无的弧度。"
        ),
        "gold_issues": [{
            "type": "ai_flavor", "quote": "他缓缓地深吸一口气，指节微微泛白",
            "severity": "major", "must_detect": True,
            "note": "动作生理套话堆砌＋空泛副词（缓缓地/微微/轻轻/若有若无）＋「勾起一抹弧度」。",
            "acceptable_quotes": ["指节微微泛白"],
        }, {
            "type": "ai_flavor", "quote": "他缓缓地抬起头，缓缓地看向窗外",
            "severity": "minor", "must_detect": False,
            "note": "v1.1 补标：同一修饰语「缓缓地」章内复现三次（深吸气/抬起头/看向窗外），构成机械节奏。",
        }, {
            "type": "ai_flavor", "quote": "眼底深处的情绪几不可察地翻涌着",
            "severity": "minor", "must_detect": False,
            "note": "v1.1 补标：空泛副词「几不可察地」修饰笼统情绪词。",
        }, {
            "type": "ai_flavor", "quote": "嘴角勾起一抹若有若无的弧度",
            "severity": "minor", "must_detect": False,
            "note": "v1.1 补标：否定对偶「若X若无X」套在「勾起弧度」套路动作上。",
        }],
    },
    {
        "id": "tc15", "category": "gold_bug", "type_focus": "ai_flavor", "difficulty": H, "chapter": 12,
        "draft_text": (
            "雪落了。巷子静了。人心也慢慢安静下来了。\n"
            "他把怀表放回口袋。他把围巾拉高一点。他把脚步放慢半拍。\n"
            "有些事，不必说破。有些路，只能一个人走。"
        ),
        "gold_issues": [{
            "type": "ai_flavor", "quote": "有些事，不必说破。有些路，只能一个人走。",
            "severity": "major", "must_detect": True,
            "note": "两处及以上同构并置（三个「他把…」）＋短句独立成句当金句＋格言式段落收尾。"
                    "表面文笔通顺，是最容易漏检的一类。",
        }, {
            "type": "ai_flavor", "quote": "他把怀表放回口袋。他把围巾拉高一点。他把脚步放慢半拍。",
            "severity": "minor", "must_detect": False,
            "note": "v1.1 补标：三连「他把…」同构并置，是可独立取证的确定性套路。",
        }, {
            "type": "ai_flavor", "quote": "人心也慢慢安静下来了",
            "severity": "minor", "must_detect": False,
            "note": "v1.1 补标：空泛副词「慢慢」堆砌氛围（run #5 实测模型报出、v1 口径下被记成误报）。",
        }],
    },
    # ---------------- clean：测误报 ----------------
    {
        "id": "tc16", "category": "clean", "type_focus": None, "difficulty": M, "chapter": 12,
        "draft_text": (
            "陆知雨把稿子改到第三遍，编辑还在催。「霜降前必须交，」她对着电话说，「我知道。」\n"
            "挂了电话，她盯着窗外那条空巷子，想起冷库门缝里看见的那一瞬——沈砚手上那片蓝。她没问过。"
            "问了，他大概也不会答。\n"
            "同一时刻，东街的修表铺里，沈砚坐着看了四十分钟。老周拆一块机芯，镊子在灯下起落。\n"
            "「今晚别出门。」老周说。\n"
            "「记得。」沈砚摸了摸口袋里的铜怀表，铜面被体温焐得温润。"
        ),
        "gold_issues": [],
        "clean_note": "全部符合设定与档案：她只看见「一片蓝」未确认霜纹；铜怀表合法；回溯提示合法。",
    },
    {
        "id": "tc17", "category": "clean", "type_focus": None, "difficulty": H, "chapter": 12,
        "draft_text": (
            "夜里十点，冷库外的巷子。\n"
            "沈砚贴着墙根走，脚下踩着薄冰——他把霜纹压到掌心，只在鞋底留了一层，走起来没有声音。\n"
            "他不碰墙。墙里有铁管。\n"
            "十点二十，他到了通风口下。十点四十，陆知雨从另一头拐进来。两个人隔着三米，谁都没先开口。"
        ),
        "gold_issues": [],
        "clean_note": "刻意埋了「墙里有铁管，所以不碰墙」这个看似可疑、实则完全合规的细节，专测 precision。",
    },
    {
        "id": "tc18", "category": "clean", "type_focus": None, "difficulty": H, "chapter": 12,
        "draft_text": (
            "「霜蚀者怕铁。」沈砚说。\n"
            "老周点头，「外头人都这么说。」\n"
            "「你也接触过？」\n"
            "「修表四十年，什么手都见过。」老周把镊子放下，没再往下说。\n"
            "沈砚没问。窗外风把招牌吹得一晃，铜铃响了一声。"
        ),
        "gold_issues": [],
        "clean_note": "老周全程含糊回避、未承认自己身份，沈砚也未点破——无 knowledge_leak，无 promise_break。",
    },
    # ---------------- decoy：看似违规实则合法 ----------------
    {
        "id": "tc19", "category": "decoy", "type_focus": None, "difficulty": H, "chapter": 12,
        "draft_text": (
            "「我知道你不是普通人。」陆知雨说。\n"
            "沈砚没接话。\n"
            "「冷库那天，门缝里，我看见你手上的东西了。我不问那是什么——我只是告诉你，我看见了。」"
        ),
        "gold_issues": [],
        "decoy_reason": "极容易被误报为 knowledge_leak。但知识状态里她「仅怀疑沈砚不普通」，"
                        "且第 11 章已埋下她看见蓝纹的记忆；她只说「看见东西了」，未使用「霜纹」概念、未确认身份，属于合法推测。",
    },
    {
        "id": "tc20", "category": "decoy", "type_focus": None, "difficulty": H, "chapter": 12,
        "draft_text": (
            "【沈砚线｜当晚十点】他推开钟楼的门，扶手上还留着那截铁链的锈痕。\n"
            "【陆知雨线｜同日傍晚】她从报社出来，天还亮着，把围巾往上拉了拉，往钟楼方向走。\n"
            "两条线要到第二天的凌晨才会交上。"
        ),
        "gold_issues": [],
        "decoy_reason": "容易被误报为 timeline 冲突（「十点」vs「天还亮着」）。但正文已显式标注线别，"
                        "两条线为同一天的不同时点交叉叙述，不构成矛盾。",
    },
]


def main() -> None:
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)), "case id 重复"
    assert len(CASES) == 20, f"应为 20 条，实际 {len(CASES)}"
    for c in CASES:
        assert c["category"] in ("gold_bug", "clean", "decoy"), c["id"]
        if c["category"] == "gold_bug":
            assert c["gold_issues"], c["id"]
        else:
            assert not c["gold_issues"], c["id"]

    # gold provider 会把每条 gold 原样当预测，而 compute_metrics 会折叠「共享 ≥8 字
    # 连续片段」的预测——如果同一 case 的两条 gold 互相高度重叠，gold 自检就会假红。
    # 在这里挡住，保证未来补标时立刻暴露。
    for c in CASES:
        quotes = [g["quote"] for g in c["gold_issues"]]
        for i in range(len(quotes)):
            for j in range(i + 1, len(quotes)):
                a, b = norm_text(quotes[i]), norm_text(quotes[j])
                assert longest_common_run(a, b) < 8, \
                    f"{c['id']} 两条 gold 共享 ≥8 字连续片段，gold 自检会被预测折叠误伤：" \
                    f"「{quotes[i]}」 vs 「{quotes[j]}」"
                assert not (len(a) >= 4 and a in b) and not (len(b) >= 4 and b in a), \
                    f"{c['id']} 两条 gold 互相包含：「{quotes[i]}」 vs 「{quotes[j]}」"

    (HERE / "context_base.json").write_text(
        json.dumps(SHARED, ensure_ascii=False, indent=2), encoding="utf-8")

    with (HERE / "consistency_v1.jsonl").open("w", encoding="utf-8") as f:
        for c in CASES:
            row = {
                "id": c["id"],
                "category": c["category"],
                "type_focus": c["type_focus"],
                "difficulty": c["difficulty"],
                "chapter": c["chapter"],
                "context": SHARED,
                "context_overrides": c.get("context_overrides", {}),
                "draft_text": c["draft_text"],
                "gold_issues": c["gold_issues"],
            }
            if c.get("clean_note"):
                row["clean_note"] = c["clean_note"]
            if c.get("decoy_reason"):
                row["decoy_reason"] = c["decoy_reason"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    bug = [c for c in CASES if c["category"] == "gold_bug"]
    n_gold = sum(len(c["gold_issues"]) for c in bug)
    n_must = sum(1 for c in bug for g in c["gold_issues"] if g.get("must_detect"))
    print(f"已生成 {len(CASES)} 条 case：gold_bug={len(bug)} / clean=3 / decoy=2")
    print(f"gold issue 共 {n_gold} 条（其中必检出 {n_must} 条）")
    print(f"→ {HERE / 'consistency_v1.jsonl'}")
    print(f"→ {HERE / 'context_base.json'}")


if __name__ == "__main__":
    main()
