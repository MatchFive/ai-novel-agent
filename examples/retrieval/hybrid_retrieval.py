"""混合检索：BM25（词面）+ 语义嵌入（词义）融合，带相关性门槛。

线上代码原样抽出，只把配置来源从项目 settings 改为内联 dataclass，使其可以独立运行。

```
查询文本 ──┬─→ 3 字 n-gram + 短语  → Python BM25 召回（项目作用域）→ 词面排名
           └─→ 原始文本            → 嵌入模型对召回池重排          → 语义排名
                                              ↓
                                    RRF 倒数排名融合 → 相关性门槛 → top-k
```

几个值得注意的设计：

1. **召回层用 Python 侧 Okapi BM25**（分词 = 字符 trigram）。线上项目早期用 SQLite FTS5
   + `bm25()`，迁到 PostgreSQL 后自带全文检索对中文分词不友好，而单项目语料规模小
   （通常 < 数千条），DB 只负责按作用域取候选、打分在进程内完成，反而更好控制。
2. **RRF 融合只用名次**，不用分数——BM25 分数和余弦相似度量纲不可比，
   倒数排名融合天然免疫尺度问题。
3. **相关性门槛**：BM25 分数在小语料/公共词场景下会趋近 0 而失去区分度，
   这时退化为「覆盖率」判据（见 `discriminative` 分支）。
4. **同步 CPU 必须挪出事件循环**：分词 + 打分是纯 CPU 循环，asyncio 单线程下会冻住整个服务
   （线上真实事故：一次手册检索触发同步嵌入，`/api/health` 都超时，而机器其它核还闲着）。
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 配置（线上从配置文件读，此处内联以便示例独立运行）
# ---------------------------------------------------------------------------


@dataclass
class RetrievalSettings:
    enabled: bool = True
    mode: str = "auto"          # off / auto / always
    semantic_enabled: bool = True
    bm25_min_ratio: float = 0.35  # 词面门槛：低于最高分该比例视为不相关
    min_cosine: float = 0.35      # 语义门槛
    rrf_k: int = 60               # RRF 平滑常数
    dense_fallback: bool = True   # 词面 0 命中时是否放语义达标者进来
    fallback_top_n: int = 0       # 无语义可用时的兜底条数（0 = 不注入）


# BM25 分数低于该值视为「无法区分」（小语料/公共词会导致分数趋近 0）
_EPS = 1e-9

# 查询词 n-gram 长度：恰好 3 字可与 trigram 语料 token 对齐
GRAM = 3

_STOP_CHARS = set("的 了 是 在 和 与 也 都 就 而 及 或 把 被 让 给 对 从 向 到 于 为 以 之 其 这 那 "
                  "有 无 不 没 很 更 最 会 能 要 可 需 将 已 又 再 还 才 只 才 便 则 若 如 因 由 "
                  "我 你 他 她 它 们 咱 谁 什 么 怎 吗 呢 吧 啊 呀 哦 嗯 "
                  "一 二 三 四 五 六 七 八 九 十 个 些 种 次 时 处 里 上 下 中 前 后 左 右 内 外 "
                  "， 。 、 ； ： ！ ？ “ ” ‘ ’ （ ） 《 》 【 】 — … · · ～ , . ; : ! ? ( ) [ ] { } "
                  "\" ' - _ / \\ | + = * & ^ % $ # @ ~ `")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------- 查询词构造
def normalize_text(text: str) -> str:
    text = _WS_RE.sub(" ", (text or "").replace("　", " ")).strip()
    return text.strip("，。、；：！？…—·,.;:!?\"'()（）【】《》[]")


def is_meaningful(term: str) -> bool:
    """整词是否含至少一个非停用字符（避免「的了」「之一」这类噪声词进查询）。"""
    return any(ch not in _STOP_CHARS for ch in term)


def ngrams(text: str, size: int = GRAM) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(len(text) - size + 1)]


def terms_from_texts(texts: Iterable[str], *, size: int = GRAM,
                     max_terms: int = 16, min_len: int = 2) -> list[str]:
    """先放完整短语，再补 3 字滑窗，截断到 max_terms；2 字词保留。"""
    out: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip()
        if len(term) < min_len or term in seen or not is_meaningful(term):
            return
        seen.add(term)
        out.append(term)

    for text in texts:
        clean = normalize_text(text)
        if not clean:
            continue
        if len(clean) <= 8:
            add(clean)
        for gram in ngrams(clean, size):
            add(gram)
    return out[:max_terms]


# ---------------------------------------------------------------- Python Okapi BM25（召回层）
def _tokenize(text: str) -> list[str]:
    """语料分词：字符 trigram（与查询 n-gram 对齐）+ 完整短语本身（≤8 字）。"""
    text = normalize_text(text)
    if not text:
        return []
    tokens = ngrams(text, GRAM)
    if len(text) <= 8:
        tokens.append(text)
    return tokens


def bm25_rank(items: Sequence[Any], terms: Sequence[str],
              text_of: Callable[[Any], str], *,
              k1: float = 1.5, b: float = 0.75,
              limit: int = 24) -> tuple[list[Any], dict[int, float]]:
    """对候选集做 Okapi BM25 排序。

    :param items: 项目作用域内的候选对象
    :param terms: 查询术语（短语 + n-gram）
    :param text_of: 取候选的比较文本（含标题，标题可在外层加权重复）
    :return: (按相关度降序的对象列表, {id: bm25 分数})
    """
    if not items or not terms:
        return [], {}
    docs = [_tokenize(text_of(it)) for it in items]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n, 1) or 1.0
    # df / idf
    df: dict[str, int] = {}
    for d in docs:
        for tok in set(d):
            df[tok] = df.get(tok, 0) + 1
    idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    scored: list[tuple[int, float]] = []
    for idx, d in enumerate(docs):
        if not d:
            continue
        tf: dict[str, int] = {}
        for tok in d:
            tf[tok] = tf.get(tok, 0) + 1
        dl = len(d)
        score = 0.0
        for t in terms:
            f = tf.get(t)
            if not f:
                continue
            score += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        if score > 0:
            scored.append((idx, score))
    scored.sort(key=lambda x: -x[1])
    picked = scored[:limit]
    return [items[i] for i, _ in picked], {int(getattr(items[i], "id")): s for i, s in picked}


# ---------------------------------------------------------------- 融合与门槛
def fuse_rrf(rank_lists: Sequence[Sequence[int]], *, k: int = 60) -> dict[int, float]:
    """倒数排名融合：score = Σ 1/(k + rank)；只用名次，免疫分数尺度不可比。"""
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, key in enumerate(ranks):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + pos + 1)
    return fused


@dataclass
class Ranked:
    key: int
    item: Any
    bm25: float = 0.0
    cosine: float | None = None
    coverage: int = 0
    lexical_ok: bool = False
    semantic_ok: bool = False
    rrf: float = 0.0

    @property
    def relevant(self) -> bool:
        return self.lexical_ok or self.semantic_ok


def count_coverage(terms: Sequence[str], text: str) -> int:
    """查询词在文本里的字面覆盖率——BM25 失去区分度时的替代判据。"""
    if not terms or not text:
        return 0
    return sum(1 for t in terms if t and t in text)


def _dense_scores(embedder, query: str, texts: list[str]) -> dict[int, float] | None:
    if embedder is None or not getattr(embedder, "enabled", False) or not query.strip():
        return None
    try:
        scored = embedder.rank_texts_scored(query, texts, top_k=len(texts))
    except Exception:  # noqa: BLE001 - 嵌入失败不能让生成失败
        return None
    if not scored:
        return None
    return {int(i): float(s) for i, s in scored}


def rank_candidates(
    *,
    pool: Sequence[Any],
    scores: dict[int, float],
    query: str,
    text_of: Callable[[Any], str],
    cfg: RetrievalSettings,
    top_k: int,
    embedder=None,
    terms: Sequence[str] | None = None,
) -> tuple[list[Any], list[Ranked]]:
    """对召回池做「语义重排 + RRF 融合 + 相关性门槛」，返回 (选中对象, 全部明细)。"""
    if not pool or top_k <= 0 or cfg.mode == "off" or not cfg.enabled:
        return [], []

    texts = [text_of(it) for it in pool]
    keys = [int(getattr(it, "id")) for it in pool]
    coverages = [count_coverage(terms or (), t) for t in texts]

    best = max(scores.values()) if scores else 0.0
    discriminative = best > _EPS

    # BM25 无法区分时（小语料/公共词），改用覆盖率排序
    if not discriminative and any(coverages):
        order = sorted(range(len(pool)), key=lambda i: -coverages[i])
        pool = [pool[i] for i in order]
        texts = [texts[i] for i in order]
        keys = [keys[i] for i in order]
        coverages = [coverages[i] for i in order]

    dense = _dense_scores(embedder, query, texts) if cfg.semantic_enabled else None
    dense_order = sorted(dense, key=lambda i: dense[i], reverse=True) if dense else []

    ranked: list[Ranked] = []
    for pos, item in enumerate(pool):
        key = keys[pos]
        bm25 = float(scores.get(key, 0.0))
        covered = coverages[pos]
        cosine = dense.get(pos) if dense else None
        if discriminative:
            lexical_ok = best > 0 and bm25 >= best * cfg.bm25_min_ratio
        else:
            lexical_ok = covered >= 1 or not any(coverages)
        semantic_ok = cosine is not None and cosine >= cfg.min_cosine
        ranked.append(Ranked(key=key, item=item, bm25=bm25, cosine=cosine,
                             coverage=covered, lexical_ok=lexical_ok,
                             semantic_ok=semantic_ok))

    fused = fuse_rrf([keys, [keys[i] for i in dense_order]], k=cfg.rrf_k)
    for r in ranked:
        r.rrf = fused.get(r.key, 0.0)

    kept = [r for r in ranked if r.relevant]
    kept.sort(key=lambda r: (r.rrf, r.coverage), reverse=True)
    return [r.item for r in kept[:top_k]], ranked


def rank_fallback(
    *,
    candidates: Sequence[Any],
    query: str,
    text_of: Callable[[Any], str],
    cfg: RetrievalSettings,
    top_k: int,
    embedder=None,
) -> tuple[list[Any], list[Ranked]]:
    """词面 0 命中时的兜底：语义达标者；无语义则按 fallback_top_n（默认 0 = 不注入）。"""
    if top_k <= 0 or cfg.mode == "off" or not cfg.enabled or not candidates:
        return [], []

    dense = _dense_scores(embedder, query, [text_of(c) for c in candidates]) if cfg.semantic_enabled else None
    if dense is not None and cfg.dense_fallback:
        picked = [(i, s) for i, s in dense.items() if s >= cfg.min_cosine]
        picked.sort(key=lambda x: x[1], reverse=True)
        ranked = [Ranked(key=int(getattr(candidates[i], "id")), item=candidates[i],
                         cosine=s, semantic_ok=True) for i, s in picked]
        return [r.item for r in ranked[:top_k]], ranked

    # 无语义可用：宁可不注入，也不拿噪声污染上下文（fallback_top_n 默认 0）
    n = min(cfg.fallback_top_n, top_k, len(candidates))
    if n <= 0:
        return [], []
    ranked = [Ranked(key=int(getattr(c, "id")), item=c, lexical_ok=True) for c in candidates[:n]]
    return [r.item for r in ranked], ranked


# ---------------------------------------------------------------- 异步包装
async def bm25_rank_async(items, terms, text_of, *, limit: int = 24):
    """`bm25_rank` 的异步版：分词 + 打分是**同步 CPU**（候选多时是明显的循环阻塞）。

    asyncio 单线程，直接在协程里跑会卡住整个服务——必须挪进线程。
    """
    import asyncio

    return await asyncio.to_thread(bm25_rank, items, terms, text_of, limit=limit)


async def rank_candidates_async(**kwargs):
    """`rank_candidates` 的异步版：语义重排走 ONNX（**同步 CPU**），必须挪出事件循环。

    线上真实事故：手册检索触发同步嵌入，进程被冻住——`/api/health` 超时、
    页面什么都加载不出来，而机器其它核还闲着（`top -p <pid>` 才看出是一个核跑满）。
    """
    import asyncio

    return await asyncio.to_thread(lambda: rank_candidates(**kwargs))
