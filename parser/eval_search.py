"""Сравнение вариантов поиска на размеченных запросах.

    python eval_search.py pool            # печатает пул кандидатов для ручной разметки
    python eval_search.py score           # считает P@10 / nDCG@10 по eval_labels.json
Метки: eval_labels.json = {"запрос": [slug релевантных компаний, ...]}.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import db
import productcenter_parser as pc
import ranking
import search

HERE = Path(__file__).parent
LABELS = HERE / "eval_labels.json"
QUERIES = ["курица", "яйцо", "мёд", "сыр", "креветки", "шоколад", "пельмени", "молоко",
           "колбаса", "хлеб", "орехи", "тушенка"]
MULTI = ["мёд липовый", "сыр моцарелла", "колбаса копчёная", "хлеб ржаной", "креветки замороженные", "орехи грецкие", "рыба соленая", "мясо индейки", "яйцо перепелиное"]
QUERIES += MULTI
LEX_GATE, COS_GATE, K = 0.3, 0.5, 10
DENSE_ONLY = 0.7


def variants(docs, query):
    """-> {имя: список индексов документов в порядке выдачи (после порога)}"""
    terms = query.split()
    out = {}
    base = np.array([ranking.relevance_stem(d, terms, pc.CATEGORIES) for d in docs])
    out["baseline"] = [i for i in np.argsort(-base, kind="stable") if base[i] >= ranking.MIN_RELEVANCE]
    
    lex = search.lexical_scores(docs, terms, pc.CATEGORIES)
    out["lexical"] = [i for i in np.argsort(-lex, kind="stable") if lex[i] >= LEX_GATE]
    
    bm25_sc = search.bm25_scores(docs, terms, pc.CATEGORIES)
    out["bm25"] = [i for i in np.argsort(-bm25_sc, kind="stable") if bm25_sc[i] > 0.0]
    
    cos = search.dense_cosine(docs, query)
    out["dense"] = [i for i in np.argsort(-cos, kind="stable") if cos[i] >= COS_GATE]
    
    fused_lex = search.rrf(lex, cos)
    keep_lex = (lex >= LEX_GATE) | (cos >= COS_GATE)
    out["hybrid_lex"] = [i for i in np.argsort(-fused_lex, kind="stable") if keep_lex[i]]
    
    fused_bm25 = search.rrf(bm25_sc, cos)
    keep_bm25 = (bm25_sc > 0.0) | (cos >= COS_GATE)
    out["hybrid_bm25"] = [i for i in np.argsort(-fused_bm25, kind="stable") if keep_bm25[i]]
    
    # Cross-Encoder (bge-reranker-v2-m3-int8) переранжирует топ-30 кандидатов из гибрида
    cand_cross = out["hybrid_bm25"][:30] if out["hybrid_bm25"] else out["hybrid_lex"][:30]
    if cand_cross:
        reranked = search.cross_encoder_rerank(query, docs, cand_cross)
        out["cross_enc"] = [idx for idx, _ in reranked]
    else:
        out["cross_enc"] = []
        
    # весь пайплайн сервиса: синонимы, роли слов, отсев непищевого, склейка дублей
    pos = {d["id"]: i for i, d in enumerate(docs)}
    res = ranking.rank(docs, {"terms": terms, "city": None, "raw": query}, pc.CATEGORIES, model="bm25_cross")
    out["service"] = [pos[r["id"]] for r in res]

    if os.environ.get("EVAL_LLM"):
        import llm_rerank
        top = cand_cross[:30]
        if top:
            new = llm_rerank.rerank(query, docs, list(lex), top_n=30)
            out["llm"] = sorted(top, key=lambda i: -new[i])
        else:
            out["llm"] = []
    return out


def ndcg(rels):
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(sum(rels), len(rels))))
    return dcg / ideal if ideal else 0.0


def main(mode):
    docs = db.all_suppliers(db.connect())
    if mode == "pool":
        for q in (sys.argv[2:] or QUERIES):
            v = variants(docs, q)
            pool = []
            for name, idx in v.items():
                for i in idx[:K]:
                    if i not in pool:
                        pool.append(i)
            print(f"\n### {q}  (в пуле {len(pool)})")
            for i in pool:
                d = docs[i]
                print(f"{d['slug']} | {d['subcategory']} | {d['name'][:38]} | {(d['description'] or '')[:70]}")
        return
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    rows = {}
    for q in QUERIES:
        rel = set(labels.get(q, []))
        for name, idx in variants(docs, q).items():
            top = [1 if docs[i]["slug"] in rel else 0 for i in idx[:K]]
            rows.setdefault(name, []).append((sum(top) / K, ndcg(top + [0] * (K - len(top))), len(idx)))
    print(f"{'вариант':14s} P@{K}   nDCG@{K}")
    for name, r in rows.items():
        print(f"{name:14s} {np.mean([x[0] for x in r]):.3f}  {np.mean([x[1] for x in r]):.3f}")
    print("\nпо запросам P@10 (варианты по порядку):")
    ref_key = "hybrid_bm25" if "hybrid_bm25" in rows else next(iter(rows))
    for j, q in enumerate(QUERIES):
        print(f"  {q:22s}", " / ".join(f"{rows[n][j][0]:.1f}" for n in rows), f" (отдано: {rows[ref_key][j][2]})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "score")
