"""Challenger B — GRU sequence encoder over trailing-16 game logs with
entity embeddings (pre-registered 2026-07-30,
BUILD_PROMPTS_model_challengers_2026-07.md §B).

Hypothesis under test (declared): NOT "deep learning beats GBDT on tabular"
— a small recurrent encoder over the raw trailing-16 player-game log, with
learned player/team/opponent-defense embeddings, captures temporal shape
(usage trajectory, role-change momentum) that fixed-span EWMs cannot, and
its representation helps THE EXISTING GBDT as features (variant B1).

Anti-leakage: a sequence for (season, week) contains ONLY games strictly
before that (season, week) in global order — the same strictly-pregame
contract as every feature in this repo.  The training task is
self-supervised (predict the NEXT game's stat vector); the encoder never
sees a line, a synthetic line, or the label the ranker is graded on.

Determinism: fixed seeds, ``torch.use_deterministic_algorithms(True)``,
single thread — same-seed retrains produce identical extracted features
(tests/test_seq_encoder.py proves it at small scale).

Fail-safe integration contract (§B2.4): features join the ranker ONLY on a
gate PASS; a missing/corrupt artifact yields NaN features which the GBDT
handles natively.  The gate verdict lives in book/seq_features_eval.json.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

SEQ_LEN = 16
STAT_CHANNELS = ["targets", "receptions", "rec_yards", "air_yards_sum",
                 "carries", "rush_yards", "pass_attempts", "pass_yards",
                 "tds_total"]
HIDDEN = 48          # <= 64 per the registered spec
PLAYER_EMB = 8       # <= 16
TEAM_EMB = 4         # <= 8
OPP_EMB = 8          # <= 16
EPOCHS = 4
BATCH = 512
LR = 1e-3
PCA_DIMS = 8
SEQ_FEATURES = [f"seq_h{i}" for i in range(PCA_DIMS)]


def _torch():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:12], 16)


# --------------------------------------------------------------------------- #
# Game-log table
# --------------------------------------------------------------------------- #
def build_game_log(pw: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Per player-game rows with stat channels + home flag + rest days,
    globally ordered.  All values are the game's OWN stats — a row is only
    ever consumed as history for strictly-later games."""
    gl = pw[["season", "week", "player_id", "team", "defteam", "role"]
            + [c for c in STAT_CHANNELS if c != "tds_total"]].copy()
    gl["tds_total"] = (pw["pass_tds"].fillna(0) + pw["rush_tds"].fillna(0)
                       + pw["rec_tds"].fillna(0))
    sched = schedules[schedules["game_type"] == "REG"][
        ["season", "week", "home_team", "away_team", "gameday"]].copy()
    home = sched.rename(columns={"home_team": "team"})[
        ["season", "week", "team", "gameday"]]
    home["home"] = 1.0
    away = sched.rename(columns={"away_team": "team"})[
        ["season", "week", "team", "gameday"]]
    away["home"] = 0.0
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])
    team_games = team_games.sort_values(["team", "gameday"], kind="mergesort")
    team_games["rest_days"] = (team_games.groupby("team")["gameday"].diff()
                               .dt.days.clip(upper=14.0))
    gl = gl.merge(team_games[["season", "week", "team", "home", "rest_days"]],
                  on=["season", "week", "team"], how="left")
    gl["home"] = gl["home"].fillna(0.5)
    gl["rest_days"] = gl["rest_days"].fillna(7.0) / 7.0
    gl["order"] = gl["season"] * 100 + gl["week"]
    return gl.sort_values(["player_id", "order"],
                          kind="mergesort").reset_index(drop=True)


def _fit_scale(gl_train: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    out = {}
    for c in STAT_CHANNELS:
        v = gl_train[c].to_numpy(dtype=float)
        out[c] = (float(np.nanmean(v)), float(np.nanstd(v) + 1e-6))
    return out


class SeqEncoder:
    """One walk-forward encoder: trained ONLY on seasons < eval_season."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.model = None
        self.scale: Dict[str, Tuple[float, float]] = {}
        self.player_index: Dict[str, int] = {}
        self.team_index: Dict[str, int] = {}
        self.meta: Dict = {}

    # ------------------------------------------------------------- internals
    def _encode_rows(self, gl: pd.DataFrame) -> np.ndarray:
        stats = np.stack([
            (gl[c].to_numpy(dtype=float) - self.scale[c][0]) / self.scale[c][1]
            for c in STAT_CHANNELS], axis=1)
        return np.nan_to_num(stats, nan=0.0)

    def _indices(self, gl: pd.DataFrame):
        player = gl["player_id"].map(lambda p: self.player_index.get(p, 0)).to_numpy()
        team = gl["team"].map(lambda t: self.team_index.get(t, 0)).to_numpy()
        opp = gl["defteam"].map(lambda t: self.team_index.get(t, 0)).to_numpy()
        return player, team, opp

    def _sequences(self, gl: pd.DataFrame, targets: bool):
        """Per player-game i (with >=1 prior game): the trailing <=SEQ_LEN
        games strictly before it.  Returns index arrays into gl plus lengths.
        With ``targets`` True, also the row's own normalized stat vector."""
        seq_rows, seq_lens, tgt_rows = [], [], []
        for _, grp in gl.groupby("player_id", sort=False):
            idx = grp.index.to_numpy()
            for j in range(1, len(idx)):
                lo = max(0, j - SEQ_LEN)
                seq_rows.append(idx[lo:j])
                seq_lens.append(j - lo)
                tgt_rows.append(idx[j])
        return seq_rows, np.array(seq_lens), (np.array(tgt_rows) if targets else None)

    # ------------------------------------------------------------------- fit
    def fit(self, gl_train: pd.DataFrame, eval_season: int) -> "SeqEncoder":
        bad = gl_train[gl_train["season"] >= eval_season]
        if len(bad):
            raise ValueError(
                f"walk-forward violation: {len(bad)} encoder-training games "
                f"at/after eval season {eval_season}")
        torch = _torch()
        torch.manual_seed(self.seed * 9973 + eval_season)
        gl = gl_train.reset_index(drop=True)
        self.scale = _fit_scale(gl)
        players = sorted(gl["player_id"].astype(str).unique().tolist())
        self.player_index = {p: i + 1 for i, p in enumerate(players)}  # 0 = UNK
        teams = sorted(set(gl["team"].astype(str)) | set(gl["defteam"].astype(str)))
        self.team_index = {t: i + 1 for i, t in enumerate(teams)}      # 0 = UNK

        stats = self._encode_rows(gl)
        player, team, opp = self._indices(gl)
        extra = gl[["home", "rest_days"]].to_numpy(dtype=float)
        seq_rows, seq_lens, tgt_rows = self._sequences(gl, targets=True)

        in_dim = len(STAT_CHANNELS) + 2 + TEAM_EMB + OPP_EMB + PLAYER_EMB
        model = _GRUNet(torch, in_dim, len(players) + 1, len(teams) + 1)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        n = len(seq_rows)
        order_rng = np.random.default_rng(self.seed * 31 + eval_season)
        for epoch in range(EPOCHS):
            perm = order_rng.permutation(n)
            total = 0.0
            for b0 in range(0, n, BATCH):
                bidx = perm[b0:b0 + BATCH]
                xb, pb, tb, ob, lens, yb = _batch(
                    torch, bidx, seq_rows, seq_lens, tgt_rows,
                    stats, extra, player, team, opp)
                opt.zero_grad()
                h = model(xb, pb, tb, ob, lens)
                loss = ((model.head(h) - yb) ** 2).mean()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(bidx)
        self.model = model
        self.meta = {
            "seed": self.seed, "eval_season": int(eval_season),
            "train_seasons": sorted(int(s) for s in gl["season"].unique()),
            "n_sequences": int(n), "n_players": len(players),
            "hidden": HIDDEN, "seq_len": SEQ_LEN, "epochs": EPOCHS,
            "final_train_mse": round(total / max(n, 1), 5),
            "task": "self-supervised next-game stat vector (no lines, no labels)",
        }
        return self

    # ------------------------------------------------------------ extraction
    def hidden_states(self, gl_all: pd.DataFrame,
                      for_rows: pd.DataFrame) -> pd.DataFrame:
        """Final hidden state for each requested (season, week, player_id):
        the sequence is that player's trailing <=16 games STRICTLY BEFORE
        (season, week) in ``gl_all``.  Rows with zero prior games get NaN
        (the GBDT's native missing-data path)."""
        torch = _torch()
        if self.model is None:
            raise RuntimeError("not fitted")
        gl = gl_all.reset_index(drop=True)
        stats = self._encode_rows(gl)
        player, team, opp = self._indices(gl)
        extra = gl[["home", "rest_days"]].to_numpy(dtype=float)
        order = (gl["season"] * 100 + gl["week"]).to_numpy()

        by_player: Dict[str, np.ndarray] = {
            pid: idx.to_numpy()
            for pid, idx in gl.groupby("player_id", sort=False).groups.items()}

        req = for_rows[["season", "week", "player_id"]]
        req = req.drop_duplicates().reset_index(drop=True)
        seqs, lens, keep = [], [], []
        for i, r in enumerate(req.itertuples(index=False)):
            idx = by_player.get(r.player_id)
            if idx is None:
                continue
            cutoff = r.season * 100 + r.week
            prior = idx[order[idx] < cutoff]
            if len(prior) == 0:
                continue
            seqs.append(prior[-SEQ_LEN:])
            lens.append(len(seqs[-1]))
            keep.append(i)
        out = np.full((len(req), HIDDEN), np.nan)
        keep_arr = np.array(keep, dtype=int)
        lens_arr = np.array(lens)
        with torch.no_grad():
            for b0 in range(0, len(seqs), BATCH):
                bidx = np.arange(b0, min(b0 + BATCH, len(seqs)))
                xb, pb, tb, ob, lb, _ = _batch(
                    torch, bidx, seqs, lens_arr, None, stats, extra,
                    player, team, opp)
                h = self.model(xb, pb, tb, ob, lb).numpy()
                out[keep_arr[bidx]] = h
        cols = pd.DataFrame(out, columns=[f"h{i}" for i in range(HIDDEN)])
        return pd.concat([req, cols], axis=1)

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> str:
        torch = _torch()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state": self.model.state_dict(), "scale": self.scale,
                    "player_index": self.player_index, "team_index": self.team_index,
                    "meta": self.meta}, path)
        with open(path + ".sha256", "w") as fh:
            fh.write(_file_sha256(path) + "\n")
        return path

    @classmethod
    def load(cls, path: str) -> "SeqEncoder":
        torch = _torch()
        sidecar = path + ".sha256"
        if os.path.exists(sidecar):
            with open(sidecar) as fh:
                expected = fh.read().strip()
            if _file_sha256(path) != expected:
                raise ValueError(f"seq encoder artifact {path} failed integrity check")
        blob = torch.load(path, weights_only=False)
        obj = cls(blob["meta"]["seed"])
        obj.scale = blob["scale"]
        obj.player_index = blob["player_index"]
        obj.team_index = blob["team_index"]
        obj.meta = blob["meta"]
        in_dim = len(STAT_CHANNELS) + 2 + TEAM_EMB + OPP_EMB + PLAYER_EMB
        obj.model = _GRUNet(torch, in_dim, len(obj.player_index) + 1,
                            len(obj.team_index) + 1)
        obj.model.load_state_dict(blob["state"])
        obj.model.eval()
        return obj


def _GRUNet(torch, in_dim: int, n_players: int, n_teams: int):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.p_emb = nn.Embedding(n_players, PLAYER_EMB)
            self.t_emb = nn.Embedding(n_teams, TEAM_EMB)
            self.o_emb = nn.Embedding(n_teams, OPP_EMB)
            self.gru = nn.GRU(in_dim, HIDDEN, num_layers=1, batch_first=True)
            self.head = nn.Linear(HIDDEN, len(STAT_CHANNELS))

        def forward(self, x, player, team, opp, lens):
            # x: (B, T, stats+2); player: (B,); team/opp: (B, T)
            B, T, _ = x.shape
            pe = self.p_emb(player).unsqueeze(1).expand(B, T, PLAYER_EMB)
            te = self.t_emb(team)
            oe = self.o_emb(opp)
            inp = torch.cat([x, te, oe, pe], dim=2)
            packed = nn.utils.rnn.pack_padded_sequence(
                inp, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            return h[-1]

    return Net()


def _batch(torch, bidx, seq_rows, seq_lens, tgt_rows, stats, extra,
           player, team, opp):
    lens = seq_lens[bidx]
    T = int(lens.max())
    B = len(bidx)
    x = np.zeros((B, T, stats.shape[1] + 2), dtype=np.float32)
    tm = np.zeros((B, T), dtype=np.int64)
    op = np.zeros((B, T), dtype=np.int64)
    pl = np.zeros(B, dtype=np.int64)
    y = (np.zeros((B, stats.shape[1]), dtype=np.float32)
         if tgt_rows is not None else None)
    for k, bi in enumerate(bidx):
        rows = seq_rows[bi]
        L = len(rows)
        x[k, :L, :stats.shape[1]] = stats[rows]
        x[k, :L, stats.shape[1]:] = extra[rows]
        tm[k, :L] = team[rows]
        op[k, :L] = opp[rows]
        pl[k] = player[rows[-1]]
        if tgt_rows is not None:
            y[k] = stats[tgt_rows[bi]]
    return (torch.tensor(x), torch.tensor(pl), torch.tensor(tm), torch.tensor(op),
            torch.tensor(lens, dtype=torch.long),
            torch.tensor(y) if y is not None else None)


def fit_pca(hidden_train: np.ndarray, seed: int):
    """PCA to PCA_DIMS, fit on TRAIN-season hidden states only."""
    from sklearn.decomposition import PCA
    mask = np.isfinite(hidden_train).all(axis=1)
    pca = PCA(n_components=PCA_DIMS, random_state=seed, svd_solver="full")
    pca.fit(hidden_train[mask])
    return pca


def project_pca(pca, hidden: np.ndarray) -> np.ndarray:
    out = np.full((len(hidden), PCA_DIMS), np.nan)
    mask = np.isfinite(hidden).all(axis=1)
    if mask.any():
        out[mask] = pca.transform(hidden[mask])
    return out


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Production integration (B2.4) — gate PASSED 2026-08-11
# --------------------------------------------------------------------------- #
# One encoder+PCA artifact, trained on all COMPLETE seasons (< its recorded
# cutoff), attaches seq_h0..7 to any candidate frame.  FAIL-SAFE by contract:
# a missing, corrupt, or digest-mismatched artifact — or any error at all —
# yields NaN feature columns, which the GBDT handles natively; the pipeline
# never crashes and never consumes a gate-failed artifact silently (the
# artifact records its gate book).
DEFAULT_ARTIFACT = "data/seq_encoder.joblib"    # matches state_store data/*.joblib

_ATTACH_CACHE: Dict = {}


def build_production_artifact(cutoff_season: int, seed: int = 7,
                              path: str = DEFAULT_ARTIFACT) -> str:
    """Train the shipped encoder on every season < cutoff_season and fit the
    PCA on those hidden states.  Live rows (>= cutoff) are strictly later
    than anything the encoder saw — the same walk-forward contract the gate
    was measured under."""
    import joblib
    from nflvalue.candidates import build_week_inputs
    inputs = build_week_inputs()
    gl = build_game_log(inputs.pw, inputs.schedules)
    enc = SeqEncoder(seed).fit(gl[gl["season"] < cutoff_season],
                               eval_season=cutoff_season)
    rows = gl[gl["season"] < cutoff_season][
        ["season", "week", "player_id"]].drop_duplicates()
    hid = enc.hidden_states(gl, rows)
    hcols = [f"h{i}" for i in range(HIDDEN)]
    pca = fit_pca(hid[hcols].to_numpy(), seed=seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({
        "state": {k: v.numpy() for k, v in enc.model.state_dict().items()},
        "scale": enc.scale, "player_index": enc.player_index,
        "team_index": enc.team_index, "meta": enc.meta,
        "pca_components": pca.components_, "pca_mean": pca.mean_,
        "gate_book": "book/seq_features_eval.json",
    }, path, compress=3)
    with open(path + ".sha256", "w") as fh:
        fh.write(_file_sha256(path) + "\n")
    return path


def _load_production_artifact(path: str):
    import joblib
    torch = _torch()
    sidecar = path + ".sha256"
    if os.path.exists(sidecar):
        with open(sidecar) as fh:
            expected = fh.read().strip()
        if _file_sha256(path) != expected:
            raise ValueError(f"seq artifact {path} failed integrity check")
    blob = joblib.load(path)
    enc = SeqEncoder(blob["meta"]["seed"])
    enc.scale = blob["scale"]
    enc.player_index = blob["player_index"]
    enc.team_index = blob["team_index"]
    enc.meta = blob["meta"]
    in_dim = len(STAT_CHANNELS) + 2 + TEAM_EMB + OPP_EMB + PLAYER_EMB
    enc.model = _GRUNet(torch, in_dim, len(enc.player_index) + 1,
                        len(enc.team_index) + 1)
    enc.model.load_state_dict({k: torch.tensor(v)
                               for k, v in blob["state"].items()})
    enc.model.eval()

    class _PCA:
        components_ = blob["pca_components"]
        mean_ = blob["pca_mean"]

        def transform(self, X):
            return (X - self.mean_) @ self.components_.T

    return enc, _PCA()


def attach_seq_features(f: pd.DataFrame, pw: Optional[pd.DataFrame],
                        path: str = DEFAULT_ARTIFACT) -> pd.DataFrame:
    """Join seq_h0..7 onto a candidate/feature frame.  Any failure — absent
    artifact, corrupt blob, missing columns, anything — stamps NaN columns
    and returns (the registered fail-safe path; tests prove it)."""
    try:
        key = (os.path.abspath(path), os.path.getmtime(path))
        if _ATTACH_CACHE.get("key") != key:
            enc, pca = _load_production_artifact(path)
            _ATTACH_CACHE.update({"key": key, "enc": enc, "pca": pca,
                                  "gl_pw_id": None, "gl": None})
        enc, pca = _ATTACH_CACHE["enc"], _ATTACH_CACHE["pca"]
        if pw is None:
            raise ValueError("no player_week table to build sequences from")
        if _ATTACH_CACHE.get("gl_pw_id") != id(pw):
            from nflvalue.ingest import load_all_schedules
            _ATTACH_CACHE["gl"] = build_game_log(pw, load_all_schedules())
            _ATTACH_CACHE["gl_pw_id"] = id(pw)
        gl = _ATTACH_CACHE["gl"]
        hid = enc.hidden_states(gl, f[["season", "week", "player_id"]])
        hcols = [f"h{i}" for i in range(HIDDEN)]
        proj = project_pca(pca, hid[hcols].to_numpy())
        tbl = hid[["season", "week", "player_id"]].copy()
        for i in range(PCA_DIMS):
            tbl[f"seq_h{i}"] = proj[:, i]
        out = f.drop(columns=[c for c in SEQ_FEATURES if c in f.columns])
        return out.merge(tbl, on=["season", "week", "player_id"], how="left")
    except Exception:
        out = f.copy()
        for c in SEQ_FEATURES:
            out[c] = np.nan
        return out
