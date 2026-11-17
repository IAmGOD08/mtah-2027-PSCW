#!/usr/bin/env python3
"""mtah/frames.py -- step 0: frame bookkeeping, reference metric, T0 ship, self-tests.

Integrates with the existing repo: games.py (GAMES[GAME] fps/pitch/
n_video_frames, asset_paths, source_frames), mtah/cache.py (load_cache ->
(players, ball, team)), mtah/level0.py (evidence, decode3), mtah/deadball.py
(detect, smart_tagging_dead), starter/baseline.py (write_submission).

Assumptions (checked at runtime where possible):
A1  cache.load_cache() returns players (n,46,2) float32 metres, ball (n,2)
    float32 metres, NaN = missing; index i == ATD frame_id i+1 == submission
    frame i (cache.py does idx = frame_id - 1). n must be 199,319, and must
    equal games n_video_frames and source_frames("tactical").
A2  team (46,): 0 = home (Barcelona), 1 = away. Any other value (unassigned
    slot, referee) belongs to neither team and is reported, not fatal.
A3  Graded blocks are inclusive: [11185, 90651] and [110808, 190197]
    (79,467 + 79,390 = 158,857; halftime gap 20,156 frames).
A4  Reference spellF1 (from the task spec; the official scorer wins if they
    disagree): X forward-filled within each block (leading X stays X and is
    not a possession); possessions = maximal H/A runs inside one block;
    same-team, one-to-one, maximum-cardinality matching over pairs with
    IoU >= thr, thr in {0.3,...,0.7}; F1 = 2TP/(n_pred+n_true), averaged.
A5  Reference macroF1 (matches evaluation/possession_metrics): per-frame F1
    for H, A, D over frames where the truth is not X; a predicted X is NOT
    forward-filled here and counts as a miss for the true class. Mean over
    classes present in truth or prediction. spellF1 does forward-fill X.
A6  Decoding is per graded block; the decoder never sees halftime or
    pre/post-match frames; D is written only outside the blocks; no X.
A7  T0 = level0 evidence (alpha 0.25, cap 2) + level0's two-state Viterbi
    arithmetic (lam 20), decoded per block. Parity with level0.evidence and
    level0.decode3 (D made unreachable) is checked on the full sequence.
A8  Smart Tagging is Sportscode XML: <instance> with <start>, <end> (seconds
    from video start), <code>, <label><group/><text/></label>; frame =
    round(s*25). A restart tag is any instance whose code or label text names
    a restart type. A team phase is any instance whose code or label text
    names exactly one club (Barcelona -> H, Athletic/Atletic -> A); N/A skipped.
A9  Alignment (primary): kick-offs detected geometrically from tracking (>= 95%
    of each team in its own half for >= 2 s, ball near the centre if visible;
    event = first frame after the formation breaks; events within 30 s merged,
    keeping the last) are matched to Smart Tagging kick-off tags. A tag marks
    an earlier moment than the formation break, so the test is CONSISTENCY of
    the offset, not its size: >= min(3, n) matches, MAD <= 25 frames, a match in
    both halves, per-half medians within 50 frames, |median| <= 250 frames.
    Team-phase agreement and restart onsets are informational only: the Team
    label names the phase's subject team, including defending phases.

CLI (from repo root):
    python -m mtah.frames selftest
    python -m mtah.frames smoke
    python -m mtah.frames diag [--deadball]
    python -m mtah.frames align [--tags PATH]     # kick-off geometry vs kick-off tags
    python -m mtah.frames tags-inventory [--tags PATH]
    python -m mtah.frames ship-t0 [--out out/T0.csv] [--no-parity]
    python -m mtah.frames crosscheck [--out selftest_csv]   # + in-process official scorer
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "starter"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402

    _CFG = GAMES[GAME]
except ImportError:  # only selftest / smoke may run without the repo
    GAME = GAMES = asset_paths = source_frames = _CFG = None

N_FRAMES = 199_319
FPS = 25
PITCH_X, PITCH_Y = 105.3, 68.0
BLOCKS = ((11185, 90651), (110808, 190197))
N_GRADED = 158_857
HALFTIME_GAP = 20_156
H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = "HADX"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
OUT_DIR = ROOT / "out"
EXPECTED = dict(ball_vis_all=0.569, ball_vis_graded=0.593, mean_players=20.5)
EXPECTED_TAGS = dict(throw_in=26, corner=12, goal_kick=10, free_kick=5)
BANDS = dict(
    possessions=(250, 450),
    median_live_s=(4.0, 15.0),
    dead_fraction=(0.15, 0.35),
    median_dead_s=(8.0, 40.0),
)
T0_PARAMS = dict(alpha=0.25, cap=2.0, lam=20.0)

assert sum(b - a + 1 for a, b in BLOCKS) == N_GRADED
assert BLOCKS[1][0] - BLOCKS[0][1] - 1 == HALFTIME_GAP


def _need_repo():
    if _CFG is None:
        raise RuntimeError(
            "games.py not importable: run from the repo root (python -m mtah.frames ...)"
        )


def check_games(verbose=True) -> dict:
    """Constants here must agree with games.py; any mismatch is fatal."""
    _need_repo()
    got = dict(
        fps=_CFG["fps"],
        pitch=tuple(_CFG["pitch"]),
        n_video_frames=_CFG.get("n_video_frames"),
        tactical_frames=source_frames("tactical", GAME),
    )
    bad = []
    if got["fps"] != FPS:
        bad.append(f"fps {got['fps']} != {FPS}")
    if not np.allclose(got["pitch"], (PITCH_X, PITCH_Y)):
        bad.append(f"pitch {got['pitch']} != {(PITCH_X, PITCH_Y)}")
    for k in ("n_video_frames", "tactical_frames"):
        if got[k] != N_FRAMES:
            bad.append(f"{k} {got[k]} != {N_FRAMES}")
    if verbose:
        print(f"[games] {GAME}: {got}")
    if bad:
        raise RuntimeError("games.py disagrees with frames.py: " + "; ".join(bad))
    return got


# --------------------------------------------------------------------------- frames
def half_index(n: int = N_FRAMES, blocks=BLOCKS) -> np.ndarray:
    """int8 (n,): -1 outside graded blocks, else block index 0/1."""
    h = np.full(n, -1, np.int8)
    for k, (a, b) in enumerate(blocks):
        h[a : b + 1] = k
    return h


def visible(xy: np.ndarray) -> np.ndarray:
    return np.isfinite(xy).all(-1)


def validate_cache(players, ball, team, n_expected: int = N_FRAMES) -> dict:
    players = np.asarray(players, np.float32)
    ball = np.asarray(ball, np.float32)
    team = np.asarray(team).astype(np.int8)
    if players.shape != (n_expected, 46, 2):
        raise ValueError(f"players shape {players.shape} != {(n_expected, 46, 2)}")
    if ball.shape != (n_expected, 2):
        raise ValueError(f"ball shape {ball.shape} != {(n_expected, 2)}")
    if team.shape != (46,):
        raise ValueError(f"team shape {team.shape} != (46,)")
    if not ((team == 0).any() and (team == 1).any()):
        raise ValueError(f"team needs both 0 and 1; got {np.unique(team)}")
    return dict(players=players, ball=ball, team=team)


def load_tracking(path=None) -> dict:
    """Load through mtah.cache.load_cache (builds the cache if missing)."""
    _need_repo()
    from mtah.cache import CACHE_PATH, load_cache

    players, ball, team = load_cache(str(path) if path else CACHE_PATH)
    return validate_cache(players, ball, team)


def cache_diagnostics(cache: dict, verbose: bool = True) -> dict:
    players, ball, team = cache["players"], cache["ball"], cache["team"]
    graded = half_index(len(ball)) >= 0
    bvis = visible(ball)
    pvis = visible(players)
    xs, ys = players[..., 0][pvis], players[..., 1][pvis]
    d = dict(
        n=int(len(ball)),
        ball_vis_all=float(bvis.mean()),
        ball_vis_graded=float(bvis[graded].mean()),
        mean_players_graded=float(pvis[graded].sum(1).mean()),
        median_home_vis=float(np.median(pvis[graded][:, team == 0].sum(1))),
        median_away_vis=float(np.median(pvis[graded][:, team == 1].sum(1))),
        other_slot_vis=float(pvis[graded][:, (team != 0) & (team != 1)].sum(1).mean()),
        x_p001=float(np.percentile(xs, 0.1)),
        x_p999=float(np.percentile(xs, 99.9)),
        y_p001=float(np.percentile(ys, 0.1)),
        y_p999=float(np.percentile(ys, 99.9)),
        slots_home=int((team == 0).sum()),
        slots_away=int((team == 1).sum()),
        slots_other=int(((team != 0) & (team != 1)).sum()),
        ball_zero_frames=int(((ball[:, 0] == 0) & (ball[:, 1] == 0)).sum()),
    )
    allx, ally = players[..., 0], players[..., 1]
    off = pvis & (
        (allx < -2) | (allx > PITCH_X + 2) | (ally < -2) | (ally > PITCH_Y + 2)
    )
    d["offpitch_share_home"] = float(
        off[graded][:, team == 0].sum() / max(pvis[graded][:, team == 0].sum(), 1)
    )
    d["offpitch_share_away"] = float(
        off[graded][:, team == 1].sum() / max(pvis[graded][:, team == 1].sum(), 1)
    )
    d["offpitch_y_gt70_share"] = float(
        (pvis & (ally > PITCH_Y + 2))[graded].sum() / max(pvis[graded].sum(), 1)
    )
    d["offpitch_y_lt_minus2_share"] = float(
        (pvis & (ally < -2))[graded].sum() / max(pvis[graded].sum(), 1)
    )
    if d["x_p999"] < 2.0 or d["y_p999"] < 2.0:
        raise ValueError("coordinates look normalised (0-1), not metres")
    checks = [
        ("n == 199,319", d["n"] == N_FRAMES),
        (
            "ball vis all ~0.569 (+-0.01)",
            abs(d["ball_vis_all"] - EXPECTED["ball_vis_all"]) <= 0.01,
        ),
        (
            "ball vis graded ~0.593 (+-0.01)",
            abs(d["ball_vis_graded"] - EXPECTED["ball_vis_graded"]) <= 0.01,
        ),
        (
            "mean players graded ~20.5 (+-1)",
            abs(d["mean_players_graded"] - EXPECTED["mean_players"]) <= 1.0,
        ),
        (
            "median per-team visible in [9,12]",
            9 <= d["median_home_vis"] <= 12 and 9 <= d["median_away_vis"] <= 12,
        ),
        ("x within [-3, 108.3]", d["x_p001"] >= -3 and d["x_p999"] <= PITCH_X + 3),
        ("y within [-3, 71]", d["y_p001"] >= -3 and d["y_p999"] <= PITCH_Y + 3),
        ("no ball at exactly (0,0)", d["ball_zero_frames"] == 0),
    ]
    d["checks"] = {name: bool(ok) for name, ok in checks}
    if verbose:
        print("[diag] cache")
        for k, v in d.items():
            if k != "checks":
                print(
                    f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}"
                )
        for name, ok in checks:
            print(f"  {'PASS' if ok else 'WARN'}  {name}")
    return d


def attack_direction(players, team, half, verbose: bool = True) -> np.ndarray:
    """int8 (2,): +1 if home attacks +x in that half. Deepest-player medians at
    both ends (goalkeeper side); raises if the two tests disagree or the sides
    do not swap at half-time."""
    signs = np.zeros(2, np.int8)
    for k in (0, 1):
        x = players[half == k][..., 0]
        hmin = np.fmin.reduce(x[:, team == 0], axis=1)
        amin = np.fmin.reduce(x[:, team == 1], axis=1)
        hmax = np.fmax.reduce(x[:, team == 0], axis=1)
        amax = np.fmax.reduce(x[:, team == 1], axis=1)
        med = lambda v: float(np.median(v[np.isfinite(v)]))
        left_home = med(hmin) < med(amin)
        right_away = (PITCH_X - med(hmax)) > (PITCH_X - med(amax))
        if left_home != right_away:
            raise RuntimeError(
                f"half {k}: goal-side tests disagree (home min {med(hmin):.1f} "
                f"away min {med(amin):.1f} home max {med(hmax):.1f} away max {med(amax):.1f})"
            )
        signs[k] = 1 if left_home else -1
        if verbose:
            print(
                f"[attack] half {k}: median deepest x home {med(hmin):.1f} away {med(amin):.1f}; "
                f"furthest x home {med(hmax):.1f} away {med(amax):.1f} -> home attacks "
                f"{'+x' if signs[k] > 0 else '-x'}"
            )
    if signs[0] != -signs[1]:
        raise RuntimeError(f"sides did not swap at half-time: {signs}")
    return signs


# --------------------------------------------------------------------------- metric
def runs(lab: np.ndarray):
    if len(lab) == 0:
        e = np.zeros(0, np.int64)
        return e, e, lab[:0]
    change = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(lab)]
    return starts, ends, lab[starts]


def ffill_x(lab: np.ndarray, blocks=BLOCKS) -> np.ndarray:
    out = lab.copy()
    for a, b in blocks:
        seg = lab[a : b + 1]
        idx = np.where(seg != X, np.arange(len(seg)), -1)
        np.maximum.accumulate(idx, out=idx)
        out[a : b + 1] = np.where(idx >= 0, seg[np.maximum(idx, 0)], X)
    return out


def possessions(lab: np.ndarray, blocks=BLOCKS) -> dict:
    """{H: (p,2), A: (q,2)} int64 [start, end_exclusive) in absolute frames."""
    res = {H: [], A: []}
    for a, b in blocks:
        s, e, v = runs(lab[a : b + 1])
        for t in (H, A):
            m = v == t
            res[t].append(np.c_[s[m] + a, e[m] + a])
    return {t: np.concatenate(res[t]).astype(np.int64) for t in (H, A)}


def _overlap_iou(T: np.ndarray, P: np.ndarray):
    if len(T) == 0 or len(P) == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0)
    lo = np.searchsorted(P[:, 1], T[:, 0], side="right")
    hi = np.searchsorted(P[:, 0], T[:, 1], side="left")
    cnt = np.maximum(hi - lo, 0)
    ti = np.repeat(np.arange(len(T)), cnt)
    pj = np.concatenate(
        [np.arange(l, h) for l, h in zip(lo, hi) if h > l] or [np.zeros(0, np.int64)]
    )
    inter = np.minimum(T[ti, 1], P[pj, 1]) - np.maximum(T[ti, 0], P[pj, 0])
    union = (T[ti, 1] - T[ti, 0]) + (P[pj, 1] - P[pj, 0]) - inter
    return ti, pj, inter / union


def spell_f1(truth, pred, blocks=BLOCKS, thresholds=THRESHOLDS, detail: bool = False):
    t, p = ffill_x(truth, blocks), ffill_x(pred, blocks)
    PT, PP = possessions(t, blocks), possessions(p, blocks)
    n_true = len(PT[H]) + len(PT[A])
    n_pred = len(PP[H]) + len(PP[A])
    pairs = {k: _overlap_iou(PT[k], PP[k]) for k in (H, A)}
    f1s = []
    for thr in thresholds:
        tp = 0
        for k in (H, A):
            ti, pj, iou = pairs[k]
            keep = iou >= thr - 1e-12
            if keep.any():
                M = csr_matrix(
                    (np.ones(int(keep.sum())), (ti[keep], pj[keep])),
                    shape=(len(PT[k]), len(PP[k])),
                )
                tp += int(
                    (maximum_bipartite_matching(M, perm_type="column") >= 0).sum()
                )
        denom = n_true + n_pred
        f1s.append(1.0 if denom == 0 else 2.0 * tp / denom)
    score = float(np.mean(f1s))
    if detail:
        return score, dict(per_threshold=f1s, n_true=n_true, n_pred=n_pred)
    return score


def macro_f1(truth, pred, blocks=BLOCKS) -> float:
    """Over graded frames with truth != X; predicted X is a miss (no fill)."""
    m = (half_index(len(truth), blocks) >= 0) & (truth != X)
    t, p = truth[m], pred[m]
    f = []
    for c in (H, A, D):
        nt, np_ = int((t == c).sum()), int((p == c).sum())
        if nt + np_:
            f.append(2.0 * int(((t == c) & (p == c)).sum()) / (nt + np_))
    return float(np.mean(f)) if f else 1.0


def shift_labels(lab, k: int, blocks=BLOCKS):
    out = lab.copy()
    for a, b in blocks:
        seg = lab[a : b + 1]
        if k > 0:
            out[a : b + 1] = np.r_[np.full(k, seg[0]), seg[:-k]]
        elif k < 0:
            out[a : b + 1] = np.r_[seg[-k:], np.full(-k, seg[-1])]
    return out


def shift_test(lab, blocks=BLOCKS, ks=(-25, -5, -1, 1, 5, 25), verbose=True) -> dict:
    """Label-free: own labels vs shifted own labels. Quantifies off-by-k cost."""
    g = half_index(len(lab), blocks) >= 0
    res = {}
    for k in ks:
        s = shift_labels(lab, k, blocks)
        res[k] = dict(
            spell=spell_f1(lab, s, blocks), agree=float((s[g] == lab[g]).mean())
        )
    if verbose:
        print("[shift] own labels vs shifted (label-free)")
        for k, r in res.items():
            print(
                f"  shift {k:+4d}: spellF1 {r['spell']:.4f}  frame agreement {r['agree']:.5f}"
            )
    return res


def label_stats(lab, blocks=BLOCKS, verbose=True, name="labels") -> dict:
    lab = ffill_x(lab, blocks)
    g = half_index(len(lab), blocks) >= 0
    P = possessions(lab, blocks)
    allp = np.concatenate([P[H], P[A]])
    dur = (allp[:, 1] - allp[:, 0]) / FPS if len(allp) else np.zeros(1)
    per_half = [
        int(sum(((P[t][:, 0] >= a) & (P[t][:, 0] <= b)).sum() for t in (H, A)))
        for a, b in blocks
    ]
    n_h, n_a = int((lab[g] == H).sum()), int((lab[g] == A).sum())
    dead_runs = []
    for a, b in blocks:
        s, e, v = runs(lab[a : b + 1])
        dead_runs += list((e - s)[v == D] / FPS)
    st = dict(
        possessions=int(len(allp)),
        per_half=per_half,
        home_share=n_h / max(n_h + n_a, 1),
        mean_live_s=float(dur.mean()),
        median_live_s=float(np.median(dur)),
        share_lt_2s=float((dur < 2.0).mean()),
        dead_fraction=float((lab[g] == D).mean()),
        dead_stretches=len(dead_runs),
        median_dead_s=float(np.median(dead_runs)) if dead_runs else 0.0,
        x_frames=int((lab[g] == X).sum()),
    )
    if verbose:
        print(f"[stats] {name} (graded frames only)")
        for k, v in st.items():
            print(f"  {k:15s} {v:.4f}" if isinstance(v, float) else f"  {k:15s} {v}")
        for key, val in (
            ("possessions", st["possessions"]),
            ("median_live_s", st["median_live_s"]),
        ):
            lo, hi = BANDS[key]
            print(f"  {'PASS' if lo <= val <= hi else 'OUT '}  {key} in [{lo},{hi}]")
        if st["dead_fraction"] > 0:
            for key in ("dead_fraction", "median_dead_s"):
                lo, hi = BANDS[key]
                print(
                    f"  {'PASS' if lo <= st[key] <= hi else 'OUT '}  {key} in [{lo},{hi}]"
                )
    return st


# --------------------------------------------------------------------------- CSV
def write_csv(lab: np.ndarray, path, submission: bool = True) -> str:
    """Writes all frames, header frame,label; returns sha256. submission=True
    enforces D outside the graded blocks (False only for synthetic truth)."""
    lab = np.asarray(lab, np.uint8)
    if len(lab) != N_FRAMES:
        raise ValueError(f"label length {len(lab)} != {N_FRAMES}")
    if lab.max() > X:
        raise ValueError("invalid label code")
    if submission and (lab[half_index() < 0] != D).any():
        raise ValueError("ungraded frames must be D")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chars = np.array(list(LABEL_CHARS))[lab]
    body = "frame,label\n" + "\n".join(f"{i},{c}" for i, c in enumerate(chars)) + "\n"
    path.write_text(body)
    return hashlib.sha256(body.encode()).hexdigest()


def _read_any(path):
    with open(path) as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [row for row in r if row]
    return (
        header,
        np.array([int(x[0]) for x in rows]),
        np.array([x[1].strip() for x in rows]),
    )


def read_csv(path) -> np.ndarray:
    header, fr, lb = _read_any(path)
    if header != ["frame", "label"]:
        raise ValueError(f"bad header {header}")
    if not np.array_equal(fr, np.arange(len(fr))):
        raise ValueError("frames must be 0..n-1 in order")
    lut = {c: i for i, c in enumerate(LABEL_CHARS)}
    return np.array([lut[c] for c in lb], np.uint8)


def writer_parity(lab, verbose=True) -> bool:
    """Our CSV vs baseline.write_submission on the same labels: identical rows."""
    import baseline as B

    with tempfile.TemporaryDirectory() as tmp:
        ours, theirs = Path(tmp) / "ours.csv", Path(tmp) / "theirs.csv"
        write_csv(lab, ours)
        B.write_submission(str(theirs), np.array(list(LABEL_CHARS))[lab])
        h1, f1, l1 = _read_any(ours)
        h2, f2, l2 = _read_any(theirs)
    ok = h1 == h2 and np.array_equal(f1, f2) and np.array_equal(l1, l2)
    if verbose:
        msg = "PASS" if ok else "FAIL"
        print(
            f"[parity] CSV vs baseline.write_submission: {msg} (header {h2}, rows {len(f2)})"
        )
        if not ok and len(f1) == len(f2):
            bad = np.flatnonzero((f1 != f2) | (l1 != l2))
            print(
                f"  first differing row {bad[:5]}; ours {f1[bad[:5]]} {l1[bad[:5]]} theirs {f2[bad[:5]]} {l2[bad[:5]]}"
            )
    return bool(ok)


# --------------------------------------------------------------------------- T0
def soft_evidence(players, ball, team, alpha=0.25, cap=2.0) -> np.ndarray:
    """Exact copy of level0.evidence: log-odds HOME has the ball, 0 = no info."""
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    d_home = d[:, team == 0].min(axis=1)
    d_away = d[:, team == 1].min(axis=1)
    ok = np.isfinite(d_home) & np.isfinite(d_away)
    z = np.zeros(len(ball), dtype=np.float32)
    z[ok] = np.clip(alpha * (d_away[ok] - d_home[ok]), -cap, cap)
    return z


def viterbi2(z, lam: float = 20.0) -> np.ndarray:
    """level0.decode3 arithmetic restricted to {H, A}, including tie-breaks:
    emissions +0.5z (H) / -0.5z (A), switch penalty lam. uint8 0=H, 1=A."""
    e = (0.5 * np.asarray(z, np.float32)).astype(np.float64).tolist()
    T = len(e)
    bp = np.zeros((T, 2), bool)  # True = predecessor was the other state
    s0 = s1 = 0.0
    for t in range(T):
        b0, f0 = s0, False
        v = s1 - lam
        if v > b0:
            b0, f0 = v, True
        b1, f1 = s0 - lam, True
        if s1 > b1:
            b1, f1 = s1, False
        s0, s1 = b0 + e[t], b1 - e[t]
        bp[t, 0], bp[t, 1] = f0, f1
    s = 0 if s0 >= s1 else 1
    path = np.empty(T, np.uint8)
    for t in range(T - 1, -1, -1):
        path[t] = s
        if bp[t, s]:
            s = 1 - s
    return path


def t0_decode(
    players, team, ball, blocks=BLOCKS, **kw
) -> tuple[np.ndarray, np.ndarray]:
    p = {**T0_PARAMS, **kw}
    z = soft_evidence(players, ball, team, p["alpha"], p["cap"])
    lab = np.full(len(ball), D, np.uint8)
    for a, b in blocks:
        lab[a : b + 1] = viterbi2(z[a : b + 1], p["lam"])
    return lab, z


def level0_parity(players, ball, team, z, lab, verbose=True) -> dict:
    """(1) soft_evidence == level0.evidence; (2) viterbi2 on the full sequence ==
    level0.decode3 with D unreachable; (3) what per-block decoding changed."""
    from mtah import level0

    z0 = level0.evidence(players, ball, team, T0_PARAMS["alpha"], T0_PARAMS["cap"])
    ev_ok = bool(np.array_equal(z, z0))
    t = time.time()
    ref = level0.decode3(
        z0, np.full(len(z0), -1e9, np.float32), lam=T0_PARAMS["lam"], lam_d=15.0
    )
    t_ref = time.time() - t
    full = viterbi2(z, T0_PARAMS["lam"])
    dec_agree = float((ref.astype(np.uint8) == full).mean())
    g = half_index() >= 0
    block_vs_full = float((full[g] == lab[g]).mean())
    r = dict(
        evidence_identical=ev_ok,
        decoder_agreement_full=dec_agree,
        level0_decode3_seconds=t_ref,
        perblock_vs_fullseq_graded_agreement=block_vs_full,
        perblock_possessions=label_stats(lab, verbose=False)["possessions"],
        fullseq_possessions=label_stats(
            np.where(g, full, D).astype(np.uint8), verbose=False
        )["possessions"],
    )
    if verbose:
        print("[parity] level0")
        for k, v in r.items():
            print(f"  {k:38s} {v}")
        print(
            f"  {'PASS' if ev_ok and dec_agree == 1.0 else 'FAIL'}  evidence identical and decoder agreement == 1.0"
        )
    return r


# --------------------------------------------------------------------------- dead-ball diagnostics
def ball_speed(ball) -> np.ndarray:
    v = np.full(len(ball), np.nan, np.float32)
    v[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    v[v > 40] = np.nan
    return v


def deadball_diagnostics(players, ball, verbose=True) -> dict:
    """Existing rule mask on graded frames, and how much of the Smart Tagging
    set-piece mask covers a moving ball (live play inside the mask)."""
    from mtah import deadball as DB

    g = half_index() >= 0
    smart = DB.smart_tagging_dead()
    t = time.time()
    dead = DB.detect(players, ball, use_smart=True)
    secs = time.time() - t
    v = ball_speed(ball)
    dr = []
    for a, b in BLOCKS:
        s, e, val = runs(dead[a : b + 1].astype(np.uint8))
        dr += list((e - s)[val == 1] / FPS)
    r = dict(
        detect_seconds=secs,
        smart_loaded=smart is not None,
        dead_all=float(dead.mean()),
        dead_graded=float(dead[g].mean()),
        dead_ungraded=float(dead[~g].mean()),
        stretches_graded=len(dr),
        median_stretch_s=float(np.median(dr)) if dr else 0.0,
    )
    if smart is not None:
        if len(smart) != len(dead):
            raise ValueError(f"smart mask length {len(smart)} != {len(dead)}")
        sm, ns = smart & g, dead & ~smart & g
        r["smart_graded"] = float(sm.sum() / g.sum())
        r["ball_visible_in_smart"] = (
            float(visible(ball)[sm].mean()) if sm.any() else float("nan")
        )
        r["ball_visible_graded"] = float(visible(ball)[g].mean())
        r["smart_frames_ball_gt5ms"] = (
            float(np.nanmean(v[sm] > 5)) if np.isfinite(v[sm]).any() else float("nan")
        )
        r["nonsmart_dead_ball_gt5ms"] = (
            float(np.nanmean(v[ns] > 5)) if np.isfinite(v[ns]).any() else float("nan")
        )
    if verbose:
        print(
            "[deadball] existing mtah.deadball.detect (rule mask), for reference only"
        )
        for k, val in r.items():
            print(f"  {k:26s} {val}")
        if smart is None:
            print(
                "  WARN  smart_tagging_dead() returned None: it swallows every exception, so detect() "
                "silently ran without tags"
            )
        lo, hi = BANDS["dead_fraction"]
        print(
            f"  {'PASS' if lo <= r['dead_graded'] <= hi else 'OUT '}  graded dead fraction in [{lo},{hi}]"
        )
    return r


# --------------------------------------------------------------------------- tags / alignment
RESTART = (
    ("throw", "throw_in"),
    ("corner", "corner"),
    ("goal kick", "goal_kick"),
    ("goalkick", "goal_kick"),
    ("free kick", "free_kick"),
    ("freekick", "free_kick"),
    ("kick off (after goal)", "kick_off_goal"),
    ("kick off (start)", "kick_off_start"),
    ("kick off", "kick_off"),
    ("kick-off", "kick_off"),
    ("kickoff", "kick_off"),
)


def default_tags_path() -> Path:
    _need_repo()
    return Path(asset_paths()["smart_tagging"])


def load_tag_restarts(path, verbose=True) -> list:
    """[(frame, kind)] from Sportscode XML; frame = round(start_seconds * 25)."""
    root = ET.parse(path).getroot()
    out, n_inst = set(), 0
    for inst in root.iter("instance"):
        n_inst += 1
        start = inst.findtext("start")
        if start is None:
            continue
        texts = [inst.findtext("code", default="")]
        texts += [lab.findtext("text", default="") for lab in inst.iter("label")]
        text = " ".join(texts).lower()
        kind = next((v for key, v in RESTART if key in text), None)
        if kind is not None:
            out.add((int(round(float(start) * FPS)), kind))
    out = sorted(out)
    if not out:
        raise ValueError(f"{path}: {n_inst} instances, no restart tags found")
    if max(f for f, _ in out) > N_FRAMES + 60 * FPS:
        raise ValueError(
            f"{path}: start times exceed video length; units are not seconds"
        )
    if verbose:
        counts = {
            k: sum(1 for _, kk in out if kk == k) for k in sorted({k for _, k in out})
        }
        print(f"[tags] {path}: {n_inst} instances, restart counts {counts}")
        for k, exp in EXPECTED_TAGS.items():
            got = counts.get(k, 0)
            print(
                f"  {'PASS' if got == exp else 'WARN'}  {k}: {got} (task statement: {exp})"
            )
    return out


def ball_onsets(
    ball, lag: int = 5, static_m: float = 0.3, move_m: float = 1.0
) -> np.ndarray:
    """Frames where a static ball starts moving (last static frame of each cluster)."""
    n = len(ball)
    p0, p1, p2 = ball[: -2 * lag], ball[lag:-lag], ball[2 * lag :]
    cand = np.zeros(n, bool)
    cand[lag : n - lag] = (np.linalg.norm(p1 - p0, axis=1) < static_m) & (
        np.linalg.norm(p2 - p1, axis=1) > move_m
    )
    return np.flatnonzero(cand & ~np.r_[cand[1:], False])


def lag_histogram(tag_frames, onsets, max_lag: int = 250, tol: int = 2) -> np.ndarray:
    """counts[L + max_lag] = #tags with an onset within +-tol frames of tag + L."""
    pad = max_lag + tol
    on = np.zeros(N_FRAMES + 2 * pad + 1, np.int32)
    on[onsets + pad] = 1
    c = np.r_[0, np.cumsum(on)]
    lags = np.arange(-max_lag, max_lag + 1)
    lo = np.clip(
        tag_frames[:, None] + lags[None, :] - tol + pad, 0, len(on) - 2 * tol - 1
    )
    return ((c[lo + 2 * tol + 1] - c[lo]) > 0).sum(0)


def restart_onset_check(tags, ball, max_lag: int = 250, verbose=True) -> dict:
    """Secondary, informational: cross-correlate restart tag starts with
    static->moving ball onsets (+-10 s, tolerance +-2 frames). Underpowered when
    the ball is missing or jittery at restarts, so it never decides pass/fail."""
    onsets = ball_onsets(ball)
    tf = np.array([f for f, _ in tags])
    lags = np.arange(-max_lag, max_lag + 1)
    hist = lag_histogram(tf, onsets, max_lag)
    coarse = int(lags[hist.argmax()])
    fine = lag_histogram(tf, onsets, max_lag, tol=0)
    win = np.abs(lags - coarse) <= 2
    vis = visible(ball)
    near = np.concatenate(
        [np.arange(max(f - 50, 0), min(f + 51, len(ball))) for f in tf]
    )
    r = dict(
        n_tags=int(len(tf)),
        n_onsets=int(len(onsets)),
        peak_lag=int(lags[win][fine[win].argmax()]),
        peak_count=int(hist.max()),
        background_median=float(np.median(hist)),
        peak_fraction_of_tags=float(hist.max() / len(tf)),
        ball_visible_within_2s_of_tags=float(vis[near].mean()),
        ball_visible_overall=float(vis.mean()),
    )
    if verbose:
        print("[align/secondary] restart tags vs ball onsets (informational only)")
        for k, v in r.items():
            print(f"  {k:32s} {v}")
    return r


TEAM_KEYS = (
    ("barcelona", H),
    ("barça", H),
    ("barca", H),
    ("atletic", A),
    ("athletic", A),
    ("bilbao", A),
)


def load_tag_teams(path, n: int = N_FRAMES, verbose=True) -> np.ndarray:
    """int8 (n,): Smart Tagging team phase per frame; -1 none / N/A / conflict."""
    root = ET.parse(path).getroot()
    home = np.zeros(n, bool)
    away = np.zeros(n, bool)
    counts = {H: 0, A: 0}
    for inst in root.iter("instance"):
        start, end = inst.findtext("start"), inst.findtext("end")
        if start is None or end is None:
            continue
        texts = [inst.findtext("code", default="")]
        texts += [lab.findtext("text", default="") for lab in inst.iter("label")]
        text = " ".join(texts).lower()
        hits = {t for key, t in TEAM_KEYS if key in text}
        if len(hits) != 1:
            continue
        t = hits.pop()
        s, e = int(round(float(start) * FPS)), int(round(float(end) * FPS))
        (home if t == H else away)[max(s, 0) : min(e + 1, n)] = True
        counts[t] += 1
    out = np.where(home & ~away, H, np.where(away & ~home, A, -1)).astype(np.int8)
    if verbose:
        g = half_index(n) >= 0
        cov = out[g] >= 0
        print(
            f"[tags] team phases: Barcelona {counts[H]}, Athletic {counts[A]}; graded coverage "
            f"{float(cov.mean()):.3f}; conflict frames {int((home & away).sum())}; Barcelona share "
            f"of covered graded frames {float((out[g] == H).sum() / max(cov.sum(), 1)):.3f}"
        )
    if counts[H] == 0 or counts[A] == 0:
        raise ValueError(f"{path}: team phases not found for both clubs ({counts})")
    return out


def _agreement(tag_team, lab, a, b, lag):
    f0, f1 = max(a, a - lag), min(b, b - lag)
    t = tag_team[f0 : f1 + 1]
    m = t >= 0
    if not m.any():
        return np.nan
    return float((lab[f0 + lag : f1 + lag + 1][m] == t[m]).mean())


def team_phase_alignment(tag_team, lab, max_lag: int = 3000, verbose=True) -> dict:
    """Primary alignment + team-mapping check, per half. Tag frame f is compared
    with label frame f + lag. peak_lag = midpoint of lags within 0.02 of the max
    agreement. Pass per half: |peak lag| <= 50 frames, peak
    agreement >= 0.60 and >= chance + 0.10, no inverted-mapping dip (fatal)."""
    res = {}
    for k, (a, b) in enumerate(BLOCKS):
        coarse_lags = np.arange(-max_lag, max_lag + 1, 5)
        ag = np.array([_agreement(tag_team, lab, a, b, L) for L in coarse_lags])
        c0 = int(coarse_lags[np.nanargmax(ag)])
        fine_lags = np.arange(c0 - 150, c0 + 151)
        fa = np.array([_agreement(tag_team, lab, a, b, L) for L in fine_lags])
        pa = float(np.nanmax(fa))
        top = fine_lags[fa >= pa - 0.02]  # level set; midpoint is robust on a flat peak
        peak = int(round((top.min() + top.max()) / 2))
        chance = float(np.nanmedian(ag[np.abs(coarse_lags) >= 1500]))
        low = float(np.nanmin(ag))
        inverted = (chance - low) > 2 * (pa - chance)
        r = dict(
            peak_lag=peak,
            peak_agreement=pa,
            agreement_lag0=_agreement(tag_team, lab, a, b, 0),
            chance=chance,
            min_agreement=low,
            inverted_mapping=bool(inverted),
            argmax_lag=int(fine_lags[np.nanargmax(fa)]),
            levelset_width_frames=int(top.max() - top.min() + 1),
        )
        r["consistent"] = bool(
            abs(peak) <= 50 and pa >= 0.60 and pa >= chance + 0.10 and not inverted
        )
        res[k] = r
        if verbose:
            print(
                f"[align/info] half {k}: team phases vs T0 labels (semantics unverified)"
            )
            for kk, v in r.items():
                print(
                    f"  {kk:18s} {v:.4f}" if isinstance(v, float) else f"  {kk:18s} {v}"
                )
    return res


def kickoff_events(
    players, team, ball, signs, min_run: int = 50, max_gap: int = 10, verbose=True
) -> np.ndarray:
    """int64 (k,): first frame after each sustained kick-off formation.
    Formation: >= 8 visible players per team, >= 95% of each team within its own
    half (+1 m tolerance), ball (if visible) within 4 m of the centre spot.
    Sides come from attack signs; frames before the halftime midpoint use half 0."""
    n = len(ball)
    cx, cy = PITCH_X / 2, PITCH_Y / 2
    mid = (BLOCKS[0][1] + BLOCKS[1][0]) // 2
    sign = np.where(np.arange(n) < mid, signs[0], signs[1])[
        :, None
    ]  # +1: home attacks +x
    x = players[..., 0]
    vis = np.isfinite(x)
    home_own = np.where(sign > 0, x < cx + 1, x > cx - 1)
    away_own = np.where(sign > 0, x > cx - 1, x < cx + 1)
    hv, av = vis & (team == 0), vis & (team == 1)
    nh, na = hv.sum(1), av.sum(1)
    fh = (home_own & hv).sum(1) / np.maximum(nh, 1)
    fa = (away_own & av).sum(1) / np.maximum(na, 1)
    bvis = visible(ball)
    bd = np.hypot(ball[:, 0] - cx, ball[:, 1] - cy)
    cond = (nh >= 8) & (na >= 8) & (fh >= 0.95) & (fa >= 0.95) & (~bvis | (bd <= 4.0))
    s, e, v = runs(cond.astype(np.uint8))
    s, e = s[v == 1], e[v == 1]
    merged = []
    for ss, ee in zip(s, e):
        if merged and ss - merged[-1][1] <= max_gap:
            merged[-1][1] = ee
        else:
            merged.append([ss, ee])
    ev = [ee for ss, ee in merged if ee - ss >= min_run]
    keep = []  # a re-formed formation within 30 s supersedes the earlier break
    for e_ in ev:
        if keep and e_ - keep[-1] <= 30 * FPS:
            keep[-1] = e_
        else:
            keep.append(e_)
    ev = np.array(keep, np.int64)
    if verbose:
        print(
            f"[kickoff] formation frames {int(cond.sum())}; formations >= {min_run} frames "
            f"(events after 30 s merge: {ev.tolist()}):"
        )
        for ss, ee in [m for m in merged if m[1] - m[0] >= min_run]:
            hk = int(half_index()[min(ee, n - 1)])
            print(
                f"  kick-off at {ee:7d} ({ee / FPS:8.2f}s)  formation {(ee - ss) / FPS:5.1f}s  "
                f"graded block {hk if hk >= 0 else '-'}"
            )
    return ev


def kickoff_alignment(tags, events, window: int = 250, verbose=True) -> dict:
    """tags: [(frame, kind)]. Consistency test of tag - event offsets (see A9)."""
    rows = []
    for f, k in sorted(tags):
        if len(events):
            j = int(np.abs(events - f).argmin())
            o = int(f - events[j])
            rows.append((int(f), k, int(events[j]), o if abs(o) <= window else None))
        else:
            rows.append((int(f), k, None, None))
    m = [(f, o) for f, _, _, o in rows if o is not None]
    off = np.array([o for _, o in m], float)
    hk = half_index()
    per_half = {
        h: [
            o
            for f, o in m
            if hk[min(max(f, 0), N_FRAMES - 1)] == h
            or (hk[min(max(f, 0), N_FRAMES - 1)] < 0 and (f < BLOCKS[1][0]) == (h == 0))
        ]
        for h in (0, 1)
    }
    med = float(np.median(off)) if len(off) else float("nan")
    mad = float(np.median(np.abs(off - med))) if len(off) else float("nan")
    hmed = {h: (float(np.median(v)) if v else None) for h, v in per_half.items()}
    need = min(3, len(rows))
    r = dict(
        n_tags=len(rows),
        n_events=int(len(events)),
        matched=int(len(off)),
        semantic_offset_frames=med,
        mad=mad,
        per_half_median=hmed,
    )
    r["pass"] = bool(
        len(off) >= need
        and mad <= 25
        and abs(med) <= 250
        and all(v is not None for v in hmed.values())
        and abs(hmed[0] - hmed[1]) <= 50
    )
    r["rows"] = rows
    if verbose:
        print(
            "[align/primary] kick-off tags vs kick-off formations (offset = tag - event, frames)"
        )
        for f, k, ev, o in rows:
            print(
                f"  tag {f:7d} {k:15s} event {ev}  "
                + (f"offset {o:+d}" if o is not None else "unmatched")
            )
        for k in (
            "n_tags",
            "n_events",
            "matched",
            "semantic_offset_frames",
            "mad",
            "per_half_median",
        ):
            print(f"  {k:24s} {r[k]}")
        print(
            f"  {'PASS' if r['pass'] else 'FAIL'}  >= {need} matched, MAD <= 25, both halves matched, "
            f"per-half medians within 50, |median| <= 250"
        )
    return r


def window_summary(
    players, team, ball, signs, f0: int, f1: int, step: int = 250, verbose=True, name=""
) -> list:
    """Per window: visible players, median player speed (m/s), ball visibility,
    share of players in own half, median ball distance to centre spot."""
    cx, cy = PITCH_X / 2, PITCH_Y / 2
    f0, f1 = max(f0, 1), min(f1, N_FRAMES)
    P = players[f0 - 1 : f1]
    sp = np.linalg.norm(np.diff(P, axis=0), axis=2) * FPS  # (f1-f0, 46)
    sp = np.where(sp > 12, np.nan, sp)
    P = P[1:]
    mid = (BLOCKS[0][1] + BLOCKS[1][0]) // 2
    rows = []
    for s in range(f0, f1, step):
        e = min(s + step, f1)
        sl = slice(s - f0, e - f0)
        x = P[sl][..., 0]
        sign = signs[0] if s < mid else signs[1]
        own_h = np.where(sign > 0, x < cx + 1, x > cx - 1)
        own_a = np.where(sign > 0, x > cx - 1, x < cx + 1)
        vis = np.isfinite(x)
        own = np.where(team == 0, own_h, own_a) & vis
        bv = visible(ball[s:e])
        bd = np.hypot(ball[s:e, 0] - cx, ball[s:e, 1] - cy)[bv]
        with np.errstate(all="ignore"):
            row = dict(
                frame=s,
                t_s=round(s / FPS, 1),
                players=float(vis.sum(1).mean()),
                speed_med=(
                    float(np.nanmedian(sp[sl]))
                    if np.isfinite(sp[sl]).any()
                    else float("nan")
                ),
                ball_vis=float(bv.mean()),
                own_half=float(own.sum() / max(vis.sum(), 1)),
                ball_to_centre=float(np.median(bd)) if len(bd) else float("nan"),
            )
        rows.append(row)
    if verbose:
        print(f"[halfstart] {name}: frames {f0}-{f1}, {step}-frame windows")
        print(
            "    frame     t_s  players  speed_med  ball_vis  own_half  ball_to_centre"
        )
        for r in rows:
            print(
                f"  {r['frame']:7d} {r['t_s']:7.1f}  {r['players']:7.1f}  {r['speed_med']:9.2f}  "
                f"{r['ball_vis']:8.2f}  {r['own_half']:8.2f}  {r['ball_to_centre']:14.1f}"
            )
    return rows


def alignment_report(tags_path, cache, lab, verbose=True) -> dict:
    restarts = load_tag_restarts(tags_path, verbose)
    ko = [(f, k) for f, k in restarts if k.startswith("kick_off")]
    signs = attack_direction(
        cache["players"], cache["team"], half_index(), verbose=False
    )
    ev = kickoff_events(
        cache["players"], cache["team"], cache["ball"], signs, verbose=verbose
    )
    prim = kickoff_alignment(ko, ev, verbose=verbose)
    starts = [f for f, k in ko if k == "kick_off_start"]
    hs = {}
    for h, (a, _) in enumerate(BLOCKS):
        near = [f for f in starts if abs(f - a) <= 5000]
        f0 = min([a] + near) - 500
        f1 = max([a] + near) + 750
        hs[h] = window_summary(
            cache["players"],
            cache["team"],
            cache["ball"],
            signs,
            f0,
            f1,
            verbose=verbose,
            name=f"block {h} start {a}, start tags {near}",
        )
    sec = restart_onset_check(restarts, cache["ball"], verbose=verbose)
    tt = load_tag_teams(tags_path, verbose=verbose)
    info = team_phase_alignment(tt, lab, verbose=verbose)
    if verbose:
        print(
            f"[align] {'PASS' if prim['pass'] else 'FAIL'}  (primary: kick-off offset consistency)"
        )
    return dict(
        primary=prim,
        halfstart=hs,
        restart_onsets=sec,
        team_phases=info,
        pass_=prim["pass"],
    )


def tags_inventory(path, top: int = 40) -> dict:
    """Distinct codes and label group:text values with counts and total seconds,
    plus how often club-named instances overlap each other."""
    root = ET.parse(path).getroot()
    codes, labels, club_codes = {}, {}, {}
    for inst in root.iter("instance"):
        try:
            dur = float(inst.findtext("end")) - float(inst.findtext("start"))
        except (TypeError, ValueError):
            continue
        code = (inst.findtext("code", default="") or "").strip()
        c = codes.setdefault(code, [0, 0.0])
        c[0] += 1
        c[1] += dur
        labs = [
            (
                (l.findtext("group", default="") or "").strip(),
                (l.findtext("text", default="") or "").strip(),
            )
            for l in inst.iter("label")
        ]
        for gt in labs:
            c = labels.setdefault(gt, [0, 0.0])
            c[0] += 1
            c[1] += dur
        text = " ".join([code] + [t for _, t in labs]).lower()
        clubs = sorted(
            {("H" if t == H else "A") for key, t in TEAM_KEYS if key in text}
        )
        if clubs:
            key = (code, "+".join(clubs))
            c = club_codes.setdefault(key, [0, 0.0])
            c[0] += 1
            c[1] += dur
    print(f"[inventory] {path}")
    print("  codes (count, total s):")
    for k, (cnt, sec) in sorted(codes.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"    {cnt:5d} {sec:9.1f}  {k!r}")
    print("  label group:text (count, total s):")
    for (g, t), (cnt, sec) in sorted(labels.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"    {cnt:5d} {sec:9.1f}  {g!r}: {t!r}")
    print("  club-named instances by code (count, total s):")
    for (code, cl), (cnt, sec) in sorted(club_codes.items(), key=lambda kv: -kv[1][0])[
        :top
    ]:
        print(f"    {cnt:5d} {sec:9.1f}  {cl:3s} {code!r}")
    return dict(codes=codes, labels=labels, club_codes=club_codes)


# --------------------------------------------------------------------------- synthetic
TAG_TEXT = dict(
    throw_in="Throw-in",
    corner="Corner",
    goal_kick="Goal Kick",
    free_kick="Free Kick",
    kick_off_start="Kick Off (Start)",
    kick_off_goal="Kick Off (After Goal)",
)


def write_sportscode_xml(tags, path, team_intervals=()):
    """tags: [(frame, kind)]; team_intervals: [(start_frame, end_frame, club_name)]."""
    root = ET.Element("file")
    allx = ET.SubElement(root, "ALL_INSTANCES")
    i = 0

    def add(s, e, code, group, text):
        nonlocal i
        i += 1
        inst = ET.SubElement(allx, "instance")
        ET.SubElement(inst, "ID").text = str(i)
        ET.SubElement(inst, "start").text = f"{s / FPS:.2f}"
        ET.SubElement(inst, "end").text = f"{e / FPS:.2f}"
        ET.SubElement(inst, "code").text = code
        lab = ET.SubElement(inst, "label")
        ET.SubElement(lab, "group").text = group
        ET.SubElement(lab, "text").text = text

    for fr, kind in tags:
        add(fr, fr + 50, "SET PIECE", "Type", TAG_TEXT[kind])
    for s, e, club in team_intervals:
        add(s, e, "PHASE", "Team", club)
    ET.ElementTree(root).write(path)


def synthetic_team_phases(truth, seed: int = 3, jitter: int = 25, na_rate: float = 0.2):
    """Noisy tag-like team phases from synthetic truth runs (smoke only)."""
    rng = np.random.default_rng(seed)
    out = []
    for a, b in BLOCKS:
        s, e, v = runs(truth[a : b + 1])
        for ss, ee, vv in zip(s + a, e + a, v):
            if vv == D or rng.random() < na_rate:
                continue
            j0, j1 = rng.integers(-jitter, jitter + 1, 2)
            out.append(
                (
                    int(ss + j0),
                    int(ee - 1 + j1),
                    "Barcelona" if vv == H else "Atletic Club",
                )
            )
    return out


def synthetic_match(seed: int = 0, n: int = N_FRAMES):
    """Synthetic cache + truth + restart tags with known answers (no match data)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    bx = np.r_[5.0, rng.uniform(15, 55, 10)]
    by = np.r_[34.0, rng.uniform(5, 63, 10)]
    amp = np.r_[2.0, np.full(10, 8.0)]
    base = np.c_[np.r_[bx, PITCH_X - bx], np.r_[by, PITCH_Y - by]]
    ph = rng.uniform(0, 2 * np.pi, (22, 2))
    px = base[:, 0] + np.r_[amp, amp] * np.sin(2 * np.pi * t / 500 + ph[:, 0])
    py = np.clip(
        base[:, 1] + 0.75 * np.r_[amp, amp] * np.sin(2 * np.pi * t / 640 + ph[:, 1]),
        0,
        PITCH_Y,
    )
    second = np.zeros(n, bool)
    second[BLOCKS[1][0] - 2000 :] = True
    px = np.where(second[:, None], PITCH_X - px, px)
    slots = np.r_[np.arange(11), 23 + np.arange(11)]
    players = np.full((n, 46, 2), np.nan, np.float32)
    players[:, slots, 0], players[:, slots, 1] = px, py
    players[rng.random((n, 46)) < 0.05] = np.nan
    team = np.r_[np.zeros(22), [-1], np.ones(23)].astype(
        np.int8
    )  # slot 22 belongs to no team
    truth = np.full(n, D, np.uint8)
    ball = np.full((n, 2), np.nan, np.float32)
    tags = []
    pos = lambda f, s: np.array([px[f, s], py[f, s]])
    for k, (a, b) in enumerate(BLOCKS):
        centre = np.array([PITCH_X / 2, PITCH_Y / 2])
        ball[a - 100 : a] = centre
        home_left = k == 0
        hx, ax_ = players[a - 100 : a, :23, 0], players[a - 100 : a, 23:, 0]
        players[a - 100 : a, :23, 0] = (
            np.minimum(hx, 50.0) if home_left else np.maximum(hx, 55.3)
        )
        players[a - 100 : a, 23:, 0] = (
            np.maximum(ax_, 55.3) if home_left else np.minimum(ax_, 50.0)
        )
        tags.append((a, "kick_off_start"))
        f, side, src = a, int(rng.integers(2)), centre
        while f <= b:
            end = min(f + max(25, int(rng.exponential(250))), b + 1)
            truth[f:end] = side
            g = f
            while g < end:
                h_end = min(g + max(15, int(rng.exponential(60))), end)
                holder = int(rng.integers(1, 11)) + 11 * side
                for j in range(g, h_end):
                    w = min(1.0, (j - g + 1) / 10)
                    ball[j] = (1 - w) * src + w * (pos(j, holder) + 0.5)
                src = ball[h_end - 1].astype(float)
                g = h_end
            f = end
            if f <= b and rng.random() < 0.25:
                dend = min(f + int(rng.uniform(150, 600)), b + 1)
                truth[f:dend] = D
                spot = np.array([rng.uniform(10, 95), 0.0])
                ball[f:dend] = src = spot
                if dend <= b:
                    tags.append((dend - 1, "throw_in"))
                f, side = dend, int(rng.integers(2))
            else:
                side = 1 - side
    vis = np.ones(n, bool)
    i = 0
    while i < n:
        i += int(rng.geometric(1 / 75))
        m = int(rng.geometric(1 / 50))
        vis[i : i + m] = False
        i += m
    ball[~vis] = np.nan
    return dict(players=players, ball=ball, team=team), truth, tags


# --------------------------------------------------------------------------- tests
def selftest() -> bool:
    print("[selftest] reference metric on analytic cases")
    n, B1, B2 = 200, ((0, 199),), ((0, 99), (120, 199))

    def lab(*segs):
        a = np.full(n, D, np.uint8)
        for s, e, c in segs:
            a[s:e] = "HADX".index(c)
        return a

    tHA = lab((0, 100, "H"), (100, 200, "A"))
    cases = [
        ("identical", spell_f1(tHA, tHA, B1), 1.0),
        (
            "late switch (IoU .74/.65)",
            spell_f1(tHA, lab((0, 135, "H"), (135, 200, "A")), B1),
            0.9,
        ),
        (
            "spurious split",
            spell_f1(
                lab((0, 200, "H")),
                lab((0, 125, "H"), (125, 135, "A"), (135, 200, "H")),
                B1,
            ),
            0.4,
        ),
        (
            "missed D merges",
            spell_f1(
                lab((0, 70, "H"), (70, 90, "D"), (90, 200, "H")), lab((0, 200, "H")), B1
            ),
            0.4,
        ),
        (
            "X inside possession free",
            spell_f1(tHA, lab((0, 50, "H"), (50, 100, "X"), (100, 200, "A")), B1),
            1.0,
        ),
        (
            "X at transition carries",
            spell_f1(tHA, lab((0, 100, "H"), (100, 135, "X"), (135, 200, "A")), B1),
            0.9,
        ),
        ("team swap", spell_f1(tHA, lab((0, 100, "A"), (100, 200, "H")), B1), 0.0),
        (
            "leading X ignored",
            spell_f1(tHA, lab((0, 10, "X"), (10, 100, "H"), (100, 200, "A")), B1),
            1.0,
        ),
        (
            "runs never span blocks",
            spell_f1(lab((0, 200, "H")), lab((0, 200, "H")), B2),
            1.0,
        ),
        (
            "ffill stops at block",
            spell_f1(lab((0, 200, "H")), lab((0, 100, "H"), (100, 200, "X")), B2),
            2 / 3,
        ),
        (
            "macroF1",
            macro_f1(lab((0, 100, "H"), (100, 150, "A"), (150, 200, "D")), tHA, B1),
            (1 + 2 / 3 + 0) / 3,
        ),
    ]
    ok = True
    for name, got, exp in cases:
        good = abs(got - exp) < 1e-9
        ok &= good
        print(
            f"  {'PASS' if good else 'FAIL'}  {name:28s} got {got:.6f} expected {exp:.6f}"
        )
    # decoder sanity: a clean switch is found; a short contrary blip is absorbed
    z = np.r_[np.full(300, 1.0), np.full(300, -1.0)].astype(np.float32)
    z[100:120] = -1.0  # 20 frames * 1.0 contrary evidence < 2 * lam 20 -> absorbed
    p = viterbi2(z, 20.0)
    good = bool((p[:300] == H).all() and (p[300:] == A).all())
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  viterbi2 switch + blip absorbed")
    print(f"[selftest] {'ALL PASS' if ok else 'FAILURES'}")
    return ok


def _official_metrics():
    """evaluation/possession_metrics as a module, or None if not in this repo."""
    d = ROOT / "evaluation"
    if not (d / "possession_metrics.py").is_file():
        return None
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    import possession_metrics as pm

    return pm


def _flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{pre}{k}."))
    else:
        out[pre[:-1]] = d
    return out


def _pick(flat, names):
    for n in names:
        for k, v in flat.items():
            if k.split(".")[-1] == n and isinstance(v, (int, float, np.floating)):
                return float(v), k
    return None, None


def official_scores(pm, truth_path, pred_path) -> dict:
    gt = pm.label_codes(pm.read_label_csv(str(truth_path)))
    pr = pm.label_codes(pm.read_label_csv(str(pred_path)))
    try:
        res = pm.evaluate(gt, pr, bootstrap_n=0)
    except Exception:
        res = pm.evaluate(gt, pr)
    flat = _flat(res)
    sp, ks = _pick(flat, ("spell_f1_mean", "spell_f1", "spellF1"))
    mf, km = _pick(flat, ("macro_f1", "macroF1", "macro"))
    if sp is None or mf is None:
        raise KeyError(
            f"could not find spell/macro keys in evaluate() result; keys: {sorted(flat)[:80]}"
        )
    return dict(spellF1=sp, macroF1=mf, keys=(ks, km))


def write_crosscheck(outdir) -> dict:
    """Synthetic full-length truth (X outside the graded blocks, like the real
    GT) and predictions, in run_evaluation.py's layout. Scores them with the
    reference metric and, when available, the official scorer in-process."""
    outdir = Path(outdir)
    _, truth, _ = synthetic_match(seed=1)
    g = half_index() >= 0
    truth = truth.copy()
    truth[~g] = X
    rng = np.random.default_rng(2)
    base = np.where(g, truth, D).astype(np.uint8)
    noisy = base.copy()
    gi = np.flatnonzero(g)
    for s0 in rng.choice(gi, 400, replace=False):
        noisy[s0 : s0 + int(rng.integers(5, 60))] = (
            1 - noisy[s0] if noisy[s0] < 2 else int(rng.integers(2))
        )
    noisy[~g] = D
    xpred = base.copy()
    xpred[gi[::7]] = X
    preds = dict(
        identical=base,
        shifted25=np.where(g, shift_labels(base, 25), D).astype(np.uint8),
        noisy=noisy,
        with_x=xpred,
    )
    write_csv(truth, outdir / "gt" / "truth.csv", submission=False)
    pm = _official_metrics()
    exp, ok = {}, True
    print(f"[crosscheck] {outdir}/gt/truth.csv vs {outdir}/results/pred_*.csv")
    print(
        f"  {'pair':10s} {'ref spell':>9s} {'ref macro':>9s} {'off spell':>9s} {'off macro':>9s}"
    )
    for k, p in preds.items():
        path = outdir / "results" / f"pred_{k}.csv"
        write_csv(p, path)
        exp[k] = dict(spellF1=spell_f1(truth, p), macroF1=macro_f1(truth, p))
        line = f"  {k:10s} {exp[k]['spellF1']:9.4f} {exp[k]['macroF1']:9.4f}"
        if pm is not None:
            off = official_scores(pm, outdir / "gt" / "truth.csv", path)
            exp[k]["official"] = off
            match = (
                abs(off["spellF1"] - exp[k]["spellF1"]) < 1e-4
                and abs(off["macroF1"] - exp[k]["macroF1"]) < 1e-4
            )
            ok &= match
            line += f" {off['spellF1']:9.4f} {off['macroF1']:9.4f}  {'PASS' if match else 'FAIL'}"
        print(line)
    (outdir / "expected.json").write_text(json.dumps(exp, indent=2, default=str))
    if pm is None:
        print(
            "  official scorer not found (evaluation/possession_metrics.py); reference only"
        )
    else:
        print(f"  official keys used: {exp['identical']['official']['keys']}")
        print(
            f"[crosscheck] {'PASS' if ok else 'FAIL'}  reference == official to 1e-4 on all pairs"
        )
    print(
        f"  CLI equivalent: python evaluation/run_evaluation.py --gt {outdir}/gt/truth.csv "
        f"--results {outdir}/results --out {outdir}/eval --no-detail"
    )
    exp["_pass"] = bool(ok)
    return exp


def smoke() -> bool:
    t0 = time.time()
    ok = selftest()
    cache, truth, tags = synthetic_match(seed=0)
    cache = validate_cache(**cache)
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "smart.xml"
        write_sportscode_xml(tags, xml_path, synthetic_team_phases(truth))
        cache_diagnostics(cache)
        sg = attack_direction(cache["players"], cache["team"], half_index())
        ok &= bool(sg[0] == 1 and sg[1] == -1)
        lab, z = t0_decode(cache["players"], cache["team"], cache["ball"])
        sha = write_csv(lab, Path(tmp) / "T0.csv")
        ok &= bool(np.array_equal(read_csv(Path(tmp) / "T0.csv"), lab))
        label_stats(lab, name="synthetic T0")
        s = spell_f1(truth, lab)
        print(
            f"[smoke] synthetic T0 vs synthetic truth: spellF1 {s:.4f} macroF1 {macro_f1(truth, lab):.4f}"
        )
        ok &= s > 0.4
        sh = shift_test(lab)
        ok &= sh[1]["spell"] >= 0.98 and sh[-1]["spell"] >= 0.98
        al = alignment_report(xml_path, cache, lab)
        ok &= al["pass_"] and al["primary"]["semantic_offset_frames"] == 0
        bad = kickoff_alignment(
            [
                (f + 1000 * (1 + i), k)
                for i, (f, k) in enumerate(
                    (f, k) for f, k in tags if k.startswith("kick_off")
                )
            ],
            kickoff_events(
                cache["players"], cache["team"], cache["ball"], sg, verbose=False
            ),
            verbose=False,
        )
        print(
            f"  {'PASS' if not bad['pass'] else 'FAIL'}  inconsistent tag offsets (40 s / 80 s) fail the check"
        )
        ok &= not bad["pass"]
    print(
        f"[smoke] sha256 {sha[:16]}  {time.time() - t0:.1f}s  -> {'PASS' if ok else 'FAIL'}"
    )
    return bool(ok)


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("smoke")
    p = sub.add_parser("diag")
    p.add_argument(
        "--deadball", action="store_true", help="also run mtah.deadball.detect (slow)"
    )
    p = sub.add_parser("align")
    p.add_argument("--tags", default=None)
    p = sub.add_parser("tags-inventory")
    p.add_argument("--tags", default=None)
    p = sub.add_parser("ship-t0")
    p.add_argument("--out", default=str(OUT_DIR / "T0.csv"))
    p.add_argument("--no-parity", action="store_true")
    p = sub.add_parser("crosscheck")
    p.add_argument("--out", default="selftest_csv")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return 0 if selftest() else 1
    if a.cmd == "smoke":
        return 0 if smoke() else 1
    if a.cmd == "tags-inventory":
        tags_inventory(a.tags or default_tags_path())
        return 0
    if a.cmd == "crosscheck":
        return 0 if write_crosscheck(a.out)["_pass"] else 1
    check_games()
    cache = load_tracking()
    if a.cmd == "diag":
        cache_diagnostics(cache)
        attack_direction(cache["players"], cache["team"], half_index())
        if a.deadball:
            deadball_diagnostics(cache["players"], cache["ball"])
        return 0
    if a.cmd == "align":
        lab, _ = t0_decode(cache["players"], cache["team"], cache["ball"])
        return (
            0
            if alignment_report(a.tags or default_tags_path(), cache, lab)["pass_"]
            else 1
        )
    if a.cmd == "ship-t0":
        t = time.time()
        diag = cache_diagnostics(cache)
        signs = attack_direction(cache["players"], cache["team"], half_index())
        lab, z = t0_decode(cache["players"], cache["team"], cache["ball"])
        g = half_index() >= 0
        assert set(np.unique(lab[g]).tolist()) <= {H, A} and (lab[~g] == D).all()
        parity = {}
        if not a.no_parity:
            parity = level0_parity(
                cache["players"], cache["ball"], cache["team"], z, lab
            )
            parity["writer"] = writer_parity(lab)
        sha = write_csv(lab, a.out)
        st = label_stats(lab, name="T0")
        sh = shift_test(lab)
        side = dict(
            module="frames.ship-t0",
            params=T0_PARAMS,
            sha256=sha,
            diag=diag,
            attack_signs=signs.tolist(),
            stats=st,
            parity=parity,
            shift={str(k): v for k, v in sh.items()},
            seconds=time.time() - t,
        )
        Path(a.out).with_suffix(".diag.json").write_text(
            json.dumps(side, indent=2, default=float)
        )
        print(f"[ship-t0] wrote {a.out}  sha256 {sha}  ({time.time() - t:.1f}s)")
        return (
            0
            if (
                a.no_parity
                or (
                    parity["evidence_identical"]
                    and parity["writer"]
                    and parity["decoder_agreement_full"] == 1.0
                )
            )
            else 1
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
