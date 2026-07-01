"""Optional semantic recall over the weekly report markdown (flag-gated).

A deliberately tiny, dependency-free TF-IDF index over ``reports/*.md`` --
enough for "which week did we flag Andrews' usage collapse?" without adding
Chroma/FAISS + an embedding model to requirements. If real embeddings are
ever wanted, swap the scorer behind :func:`search` and keep the interface;
the flag (config ``rag.vectorstore_enabled``, default false) already gates
callers either way.

The reports themselves are the corpus ON PURPOSE: they're the auditable,
already-published artifacts (PROP_SHORTLISTER_SPEC.md §6 "doubles as the RAG
context pack") -- searching them can only ever recall what was actually said.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from .. import config as cfgmod

REPORTS_DIR = os.path.join(cfgmod.ROOT, "reports")
_TOKEN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def enabled(cfg: Optional[Dict] = None) -> bool:
    cfg = cfg or cfgmod.load_config()
    return bool((cfg.get("rag") or {}).get("vectorstore_enabled", False))


def build_index(reports_dir: str = REPORTS_DIR) -> Dict:
    """Index every report: {docs: {path: Counter}, df: Counter, n_docs}."""
    docs: Dict[str, Counter] = {}
    if os.path.isdir(reports_dir):
        for fn in sorted(os.listdir(reports_dir)):
            if fn.endswith(".md"):
                path = os.path.join(reports_dir, fn)
                with open(path, errors="ignore") as f:
                    docs[path] = Counter(_tokens(f.read()))
    df = Counter()
    for c in docs.values():
        df.update(c.keys())
    return {"docs": docs, "df": df, "n_docs": len(docs)}


def search(query: str, index: Optional[Dict] = None, k: int = 3,
           reports_dir: str = REPORTS_DIR, cfg: Optional[Dict] = None) -> Dict:
    """TF-IDF cosine-ish scoring. Respects the enable flag unless an explicit
    index is passed (tests)."""
    if index is None:
        if not enabled(cfg):
            return {"status": "disabled",
                    "hint": "set config.json rag.vectorstore_enabled=true to index reports/"}
        index = build_index(reports_dir)
    if not index["docs"]:
        return {"status": "empty", "results": []}

    n = max(index["n_docs"], 1)
    q = Counter(_tokens(query))
    scored = []
    for path, doc in index["docs"].items():
        s = 0.0
        for term, qf in q.items():
            if term in doc:
                idf = math.log(1.0 + n / (1 + index["df"][term]))
                s += qf * (1.0 + math.log(1 + doc[term])) * idf * idf
        if s > 0:
            scored.append((s, path))
    scored.sort(reverse=True)
    results = []
    for s, path in scored[:k]:
        with open(path, errors="ignore") as f:
            text = f.read()
        snippet = _best_snippet(text, q)
        results.append({"path": path, "score": round(s, 3), "snippet": snippet})
    return {"status": "ok", "results": results}


def _best_snippet(text: str, q: Counter, width: int = 240) -> str:
    """The line window containing the most query terms."""
    lines = text.splitlines()
    best_i, best_hits = 0, -1
    for i, line in enumerate(lines):
        toks = set(_tokens(line))
        hits = sum(1 for t in q if t in toks)
        if hits > best_hits:
            best_i, best_hits = i, hits
    window = " ".join(lines[max(0, best_i - 1):best_i + 2]).strip()
    return (window[:width] + "…") if len(window) > width else window
