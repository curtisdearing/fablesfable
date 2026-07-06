"""Schema-aware, READ-ONLY natural-language -> SQL over the prop warehouse.

Safety model (all enforced in :func:`validate_sql`, independent of whatever
generates the SQL -- a rule-based translator today, optionally a real LLM
later behind the same interface):

  * SELECT-only whitelist: the statement must start with SELECT; every table
    referenced after FROM/JOIN must be in :data:`WHITELIST_TABLES` (so no
    ``sqlite_master``, no ``api_credits`` ledger tampering reads-into-writes,
    and obviously no DDL/DML -- INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/
    PRAGMA/ATTACH/VACUUM/REPLACE are rejected as words anywhere).
  * single statement, no comments (comment syntax is how injections hide).
  * ROW CAP: execution wraps the query as ``SELECT * FROM (<sql>) LIMIT cap``
    so no generator mistake can dump the warehouse.
  * the ANSWER is composed ONLY from returned rows (deterministic summarizer
    by default; an LLM answerer would receive rows + question, nothing else,
    and cannot add numbers that aren't in the rows).

Returns {question, sql, rows, answer, citations} -- the SQL and the table
names are always shown so every answer is auditable.

CLI:  python3 -m nflvalue.rag.nl2sql "average CLV so far"
"""

from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional, Protocol

from .. import db as dbmod

WHITELIST_TABLES = {
    "player_week", "opp_pos_def", "projections", "prop_backtest",
    "manual_notes", "leans", "lines", "clv",
    "lean_outcomes", "model_adjustments", "context_ledger", "candidate_aggregates",
}
FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "trigger", "reindex", "analyze",
}
DEFAULT_ROW_CAP = 200


class SQLValidationError(ValueError):
    """The generated SQL violated the read-only contract. Always fatal."""


class NL2SQLClient(Protocol):  # pragma: no cover - interface only
    def generate_sql(self, question: str, schema_doc: str) -> str: ...


def schema_doc() -> str:
    """The whitelisted schema, verbatim from db.py's DDL (given to any LLM
    generator so it can only ever see what it may query)."""
    parts = [ddl.strip() for name, ddl in dbmod.SCHEMA.items() if name in WHITELIST_TABLES]
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Validation (the actual security boundary)
# --------------------------------------------------------------------------- #
def validate_sql(sql: str) -> str:
    """Return the cleaned SQL or raise SQLValidationError. Never modifies
    semantics -- only strips whitespace/trailing semicolon."""
    if not sql or not sql.strip():
        raise SQLValidationError("empty SQL")
    s = sql.strip()
    if "--" in s or "/*" in s:
        raise SQLValidationError("comments are not allowed in generated SQL")
    if s.endswith(";"):
        s = s[:-1].rstrip()
    if ";" in s:
        raise SQLValidationError("multiple statements are not allowed")
    if not re.match(r"(?is)^\s*select\b", s):
        raise SQLValidationError("only SELECT statements are allowed")
    words = set(re.findall(r"[a-zA-Z_]+", s.lower()))
    bad = words & FORBIDDEN
    if bad:
        raise SQLValidationError(f"forbidden keyword(s): {sorted(bad)}")
    # belt-and-suspenders: never allow sqlite internals, however referenced.
    if re.search(r"(?i)\bsqlite_\w+", s):
        raise SQLValidationError("sqlite_* internal tables are never queryable")
    # EVERY table referenced by any FROM/JOIN clause must be whitelisted --
    # including comma-separated implicit cross-joins, which are past the reach
    # of a "first identifier after the keyword" match. Capture the whole clause
    # up to the next SQL keyword / clause terminator, then validate each
    # comma-split identifier. This tool never needs comma-joins, so a bare comma
    # inside a FROM/JOIN clause is rejected outright.
    saw_from = False
    clause_re = re.compile(
        r"(?is)\b(?:from|join)\s+(.*?)"
        r"(?=\b(?:where|group|order|limit|union|having|on|inner|left|right|"
        r"full|cross|outer|natural|join)\b|;|$)")
    for clause in clause_re.findall(s):
        saw_from = True
        clause = clause.strip().rstrip(")")
        # split on commas so implicit cross-joins can't smuggle a second table
        for part in clause.split(","):
            part = part.strip().strip("()")
            if not part:
                raise SQLValidationError("empty table reference in FROM/JOIN clause")
            # first token is the table identifier (an alias may follow)
            m = re.match(r"([a-zA-Z_][\w]*)", part)
            if not m:
                raise SQLValidationError(
                    f"unrecognized table reference {part!r} in FROM/JOIN clause")
            tbl = m.group(1)
            if tbl.lower() not in WHITELIST_TABLES:
                raise SQLValidationError(
                    f"table {tbl!r} is not in the read whitelist {sorted(WHITELIST_TABLES)}")
    if not saw_from or not re.search(r"(?is)\bfrom\s+", s):
        raise SQLValidationError("SELECT without FROM is not a warehouse query")
    return s


def execute(conn, sql: str, row_cap: int = DEFAULT_ROW_CAP) -> List[Dict]:
    """Validate, hard-cap, run. The cap is structural (outer LIMIT), not advisory."""
    clean = validate_sql(sql)
    capped = f"SELECT * FROM ({clean}) LIMIT {int(row_cap)}"
    df = dbmod.query_df(conn, capped)
    return df.to_dict("records")


# --------------------------------------------------------------------------- #
# Rule-based translator (deterministic default; LLM pluggable via NL2SQLClient)
# --------------------------------------------------------------------------- #
class RuleBasedNL2SQL:
    """Canned patterns for the questions this warehouse actually gets asked.
    Anything unrecognized falls back to 'recent leans' -- visibly, in the SQL."""

    def generate_sql(self, question: str, schema_doc: str) -> str:  # noqa: ARG002
        q = question.lower()
        season = self._first(re.findall(r"\b(20\d{2})\b", q))
        week = self._first(re.findall(r"\bweek\s+(\d{1,2})\b", q))
        where = []
        if season:
            where.append(f"season = {int(season)}")
        if week:
            where.append(f"week = {int(week)}")
        wh = (" WHERE " + " AND ".join(where)) if where else ""
        wh_and = (" AND " + " AND ".join(where)) if where else ""

        if "clv" in q or "closing line" in q:
            return ("SELECT COUNT(*) AS n, ROUND(AVG(clv_prob), 5) AS avg_clv_prob, "
                    "ROUND(AVG(CASE WHEN clv_prob > 0 THEN 1.0 ELSE 0.0 END), 4) AS positive_rate, "
                    "ROUND(AVG(point_moved), 3) AS avg_point_move "
                    f"FROM clv{wh}")
        if "screened" in q or "screen count" in q:
            return ("SELECT game_id, MAX(screened_n) AS screened_n, COUNT(*) AS leans "
                    f"FROM leans{wh} GROUP BY game_id ORDER BY game_id")
        if "voided" in q or "void" in q:
            return ("SELECT name, market, side, line, void_reason FROM leans "
                    f"WHERE status = 'voided'{wh_and} ORDER BY season, week")
        if "lean" in q or "pick" in q or "shortlist" in q:
            return ("SELECT name, market, side, line, line_source, composite, edge, "
                    "confidence_comp, screened_n, status, reason "
                    f"FROM leans{wh} ORDER BY composite DESC")
        for market, kw in (("receiving_yards", "receiving"), ("rushing_yards", "rushing"),
                           ("passing_yards", "passing"), ("receptions", "reception")):
            if kw in q:
                col = {"receiving_yards": "rec_yards", "rushing_yards": "rush_yards",
                       "passing_yards": "pass_yards", "receptions": "receptions"}[market]
                return (f"SELECT player_name, team, week, {col} "
                        f"FROM player_week{wh} ORDER BY {col} DESC")
        if "backtest" in q or "accuracy" in q or "mae" in q:
            return ("SELECT market, sample_bucket, n, mae, rmse, corr FROM prop_backtest "
                    "WHERE calibration_bucket = 'overall' ORDER BY market, sample_bucket")
        return ("SELECT season, week, name, market, side, line, composite, status "
                f"FROM leans{wh} ORDER BY created_at DESC")

    @staticmethod
    def _first(matches: List[str]) -> Optional[str]:
        return matches[0] if matches else None


def summarize_rows(question: str, sql: str, rows: List[Dict]) -> str:
    """Deterministic answer from ONLY the returned rows. No row, no claim."""
    if not rows:
        return ("The query returned no rows — there is no data matching this question "
                "in the warehouse (nothing is inferred beyond that).")
    n = len(rows)
    first = rows[0]
    if n == 1 and len(first) <= 6:
        kv = "; ".join(f"{k}={v}" for k, v in first.items())
        return f"1 row: {kv}."
    preview_keys = list(first.keys())[:5]
    lines = []
    for r in rows[:3]:
        lines.append(", ".join(f"{k}={r.get(k)}" for k in preview_keys))
    return (f"{n} row(s) returned (capped). First {min(3, n)}: " + " | ".join(lines)
            + ". Full rows accompany this answer.")


def run_query(question: str, conn=None, client: Optional[NL2SQLClient] = None,
              row_cap: int = DEFAULT_ROW_CAP) -> Dict:
    conn = conn or dbmod.connect()
    client = client or RuleBasedNL2SQL()
    sql = client.generate_sql(question, schema_doc())
    clean = validate_sql(sql)          # validate BEFORE touching the DB, whoever wrote it
    rows = execute(conn, clean, row_cap=row_cap)
    tables = sorted({t.lower() for t in re.findall(r"(?is)\b(?:from|join)\s+([a-zA-Z_][\w]*)", clean)})
    return {
        "question": question,
        "sql": clean,
        "rows": rows,
        "answer": summarize_rows(question, clean, rows),
        "citations": {"tables": tables, "row_count": len(rows), "row_cap": row_cap},
    }


def main() -> None:  # pragma: no cover - thin CLI
    if len(sys.argv) < 2:
        print('usage: python3 -m nflvalue.rag.nl2sql "your question"')
        raise SystemExit(2)
    res = run_query(" ".join(sys.argv[1:]))
    print(f"Q: {res['question']}\nSQL: {res['sql']}\n"
          f"Tables: {', '.join(res['citations']['tables'])} · rows: {res['citations']['row_count']}")
    for r in res["rows"][:10]:
        print("  ", r)
    print(f"A: {res['answer']}")


if __name__ == "__main__":
    main()
