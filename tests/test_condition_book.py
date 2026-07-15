"""Condition book: schema, shrinkage sanity, and the display-only panel contract."""
import os

import pandas as pd
import pytest

from nflvalue import condition_book as cb


@pytest.fixture()
def book():
    df = cb.load_book(refresh=True)
    if df.empty:
        pytest.skip("book not built (run refresh_condition_book.py)")
    return df


def test_schema(book):
    for col in ["player_id", "condition", "stat", "n", "raw_over", "own_base",
                "shrunk_over", "edge_pp", "flag"]:
        assert col in book.columns, col
    assert (book["n"] >= 3).all()
    assert book["shrunk_over"].between(0, 1).all()


def test_shrinkage_pulls_toward_base(book):
    # shrunk rate must sit between raw rate and the player's own base
    lo = book[["raw_over", "own_base"]].min(axis=1) - 1e-9
    hi = book[["raw_over", "own_base"]].max(axis=1) + 1e-9
    assert book["shrunk_over"].between(lo, hi).all()


def test_edges_for_orders_by_magnitude(book):
    pid = book.loc[book["flag"], "player_id"].iloc[0]
    d = cb.edges_for(pid)
    mags = d["edge_pp"].abs().tolist()
    assert mags == sorted(mags, reverse=True)


def test_panel_items_contract(book):
    pid = book.loc[book["flag"], "player_id"].iloc[0]
    items = cb.panel_items({"player_id": pid})
    assert isinstance(items, list) and len(items) <= 3
    assert all(i.startswith("condition book:") for i in items)
    # unknown player and missing player_id stay silent, never raise
    assert cb.panel_items({"player_id": "no-such-player"}) == []
    assert cb.panel_items({}) == []


def test_panel_is_display_only():
    """The module must not import composite/model/shortlist (no score path)."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(cb))
    banned = {"composite", "model", "shortlist", "ml_ranker", "oddsmath"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not (set(node.module.split(".")) & banned)
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not (set(a.name.split(".")) & banned)
