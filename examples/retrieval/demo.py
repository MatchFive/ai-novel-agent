"""混合检索示例：用中文小说记忆库演示 BM25 召回 + 语义重排 + 相关性门槛。

直接运行：`python demo.py`（无需 API Key、无需外部依赖）

场景：一本玄幻小说的记忆库。每章生成前，检索引擎要从全库记忆里挑出
**这一章真正相关**的几条注入提示词——挑错 = 上下文污染（模型会凭空编造关联），
挑漏 = 设定崩坏。所以「注入什么」比「召回多少」更要紧。

线上查询文本是**章节大纲 / 情节描述**这类长文本，不是关键词——
这也是为什么下面的查询都是一句话。短关键词在 trigram 体系下几乎召回不到东西
（见文件末尾的对照实验）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from hybrid_retrieval import (
    RetrievalSettings,
    bm25_rank,
    rank_candidates,
    terms_from_texts,
)


@dataclass
class Memory:
    id: int
    kind: str      # canon 世界观铁律 / experience 角色经历 / promise 未回收伏笔
    text: str


MEMORIES = [
    Memory(1, "canon",      "霜降之城终年积雪，城墙由寒铁铸成，城中四季不分明。"),
    Memory(2, "experience", "沈昭在第十二章被逐出家族，自此不再使用沈氏剑法。"),
    Memory(3, "promise",    "老者在第三章交给沈昭一枚断裂的玉佩，至今未说明来历。"),
    Memory(4, "experience", "沈昭十九岁，左眉有一道旧疤，是幼年坠崖所留。"),
    Memory(5, "canon",      "城中禁地「白塔」不许任何人靠近，塔顶常年有钟声。"),
    Memory(6, "experience", "苏晚在第二十章发现族谱被篡改，生母一栏被人抹去。"),
    Memory(7, "promise",    "苏晚生母的真实身份始终未被揭示，她本人亦不知情。"),
    Memory(8, "canon",      "寒铁遇火属性灵力会变脆，这是城防最大的隐患。"),
    Memory(9, "experience", "苏晚与沈昭在第九章于城南酒肆初次相遇。"),
    Memory(10, "canon",     "城中有三条灵脉，分别对应冰、风、暗三系修士。"),
    Memory(11, "experience", "沈昭曾把玉佩拿到当铺估价，掌柜看了一眼便脸色大变，拒收。"),
    Memory(12, "canon",     "断裂玉佩上刻着冰系灵纹，与城中三条灵脉的冰脉同源。"),
    Memory(13, "promise",   "白塔钟声只在每月朔日响起，与城外某人的约定有关。"),
    Memory(14, "experience", "沈昭幼年坠崖之地就在白塔附近，他对此已毫无记忆。"),
    Memory(15, "experience", "城防军曾在第十五章因寒铁脆化，塌了一段城墙。"),
]


class BigramEmbedder:
    """模拟嵌入模型。

    真实项目用本地 ONNX 语义模型，接口一致：`rank_texts_scored(query, texts, top_k)
    -> [(index, score)]`，失败的返回 None 走词面兜底。
    这里用字符 bigram 的查询覆盖率代替语义向量，只为让示例可以独立运行。
    """

    enabled = True
    _STRIP = re.compile(r"[\s，。、；：！？…—·「」【】（）《》]+")

    def _bigrams(self, text: str) -> set[str]:
        t = self._STRIP.sub("", text or "")
        return {t[i:i + 2] for i in range(len(t) - 1)}

    def rank_texts_scored(self, query: str, texts: list[str], top_k: int | None = None):
        q = self._bigrams(query)
        if not q:
            return []
        out = []
        for i, t in enumerate(texts[: top_k or len(texts)]):
            d = self._bigrams(t)
            if not d:
                continue
            out.append((i, len(q & d) / len(q)))
        return sorted(out, key=lambda x: -x[1])


CFG = RetrievalSettings()
TOP_K = 3
EMBEDDER = BigramEmbedder()


def search(query: str, *, verbose: bool = True) -> list[Memory]:
    terms = terms_from_texts([query], max_terms=40)

    # L1 · 词面召回：从项目作用域内的候选里取回一批（线上这一步只捞候选，不判相关）
    recalled, scores = bm25_rank(MEMORIES, terms, lambda m: m.text, limit=len(MEMORIES))

    # L2 · 语义重排 + RRF 融合 + 相关性门槛
    picked, ranked = rank_candidates(
        pool=recalled,
        scores=scores,
        query=query,
        text_of=lambda m: m.text,
        cfg=CFG,
        top_k=TOP_K,
        embedder=EMBEDDER,
        terms=terms,
    )

    if verbose:
        print(f"\n查询：{query}")
        if not recalled:
            print(f"  L1 召回 0 条 →  直接不注入（全库 {len(MEMORIES)} 条，一条都不给模型）")
            return picked
        print(f"  L1 召回 {len(recalled)} 条：{['#' + str(m.id) for m in recalled]}")
        print(f"  L2 精排后注入 {len(picked)} 条：")
        for m in picked:
            r = next(r for r in ranked if r.key == m.id)
            cos = f"{r.cosine:.2f}" if r.cosine is not None else "  - "
            print(f"    ✓ [{m.kind:10s}] bm25={r.bm25:5.2f}  cos={cos}  {m.text}")
        dropped = [r for r in ranked if r.item not in picked]
        if dropped:
            print(f"    其余 {len(dropped)} 条未入选（RRF 排名靠后，或未过相关性门槛）")
    return picked


if __name__ == "__main__":
    print("=" * 80)
    print(f"混合检索演示 · 中文小说记忆库（全库 {len(MEMORIES)} 条）")
    print("=" * 80)
    print("查询文本模拟「本章大纲 / 情节描述」——线上就是这么使用的。")
    print("检索要做的事：从全库挑出**这一章真正相关**的几条注入提示词。")

    search("沈昭取出老者交给他的那枚断裂的玉佩，玉佩上刻着冰系灵纹，他一直不解其意。")
    search("苏晚生母的真实身份始终成谜，她在族谱中查到生母一栏被人抹去，决心弄清究竟是谁。")
    search("沈昭被逐出家族后，为何再也不肯使用沈氏剑法？")
    search("城防军发现寒铁城墙脆化，火属性灵力一碰就裂。")
    search("今天天气不错，适合出门散步。")   # 无关查询：应当一条都不注入

    print("\n" + "=" * 80)
    print("要点")
    print("=" * 80)
    print("1. 召回靠词面（BM25 + 字符 trigram），中文无需分词器，小语料完全够用。")
    print("2. RRF 融合只用名次，不用分数——BM25 分数与余弦相似度量纲不可比。")
    print("3. 相关性门槛比召回率更关键：无关查询宁可返回空，也不拿噪声污染上下文。")
    print("4. 精排只能重排召回池里的东西——**召回池决定上限**，所以池子怎么捞很重要")

    print("\n对照实验：查询文本长度对召回的影响")
    print("-" * 80)
    for q in ["沈昭玉佩", "沈昭带着那枚断裂的玉佩，想起老者当年的叮嘱"]:
        terms = terms_from_texts([q], max_terms=16)
        recalled, _ = bm25_rank(MEMORIES, terms, lambda m: m.text, limit=len(MEMORIES))
        print(f"  {q:<26} 查询词 {len(terms):>2} 个 → 召回 {len(recalled)} 条 "
              f"{['#' + str(m.id) for m in recalled]}")
    print("\n  短关键词召回不到，是因为 2 字人名 / 词在 3 字 trigram 语料里没有对应 token。")
    print("  线上把「章节大纲 / 情节描述」作为查询文本，天然规避了这个问题。")
