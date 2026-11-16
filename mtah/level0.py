"""Level 0: nearest-player evidence + two-state Viterbi decoder.

Evidence: for each frame with a visible ball, compute the log-odds that the
nearest home player is closer than the nearest away player. This is a soft
score, not a hard label, so the decoder can weigh it against the cost of
switching.

Decoder: a two-state HMM with a switching penalty. Frames with no ball get
zero evidence, so the state carries through the gap. This is different from
majority-vote smoothing, which treats a gap as "nobody has it" and can flip
the label across it.

Dead ball is not handled here. That comes in a later module. For now, every
frame gets H or A, which is wrong during stoppages but is the right starting
point for the possession decoder.
"""

import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES  # noqa: E402
import baseline as B  # noqa: E402
from mtah.cache import load_cache  # noqa: E402

OUT_DIR = os.path.join(ROOT, "out")

HOME, AWAY = 0, 1
from mtah.deadball import detect as detect_deadball


def evidence(players, ball, team, alpha=0.25, cap=2.0):
    """Per-frame log-odds that HOME has the ball. 0 = no information.

    alpha controls how strongly distance difference maps to log-odds.
    cap prevents a single frame from dominating the decoder.
    """
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    d_home = d[:, team == HOME].min(axis=1)
    d_away = d[:, team == AWAY].min(axis=1)
    ok = np.isfinite(d_home) & np.isfinite(d_away)
    z = np.zeros(len(ball), dtype=np.float32)
    z[ok] = np.clip(alpha * (d_away[ok] - d_home[ok]), -cap, cap)
    return z


def decode(z, lam=10.0):
    """Viterbi over {H, A} with switching cost lam.

    z is the per-frame log-odds favouring home. lam is the cost of switching
    state. Frames with z=0 carry the previous state through, which is what
    makes this better than smoothing for gaps.
    """
    n = len(z)
    back_h = np.zeros(n, dtype=bool)
    back_a = np.zeros(n, dtype=bool)
    score_h = score_a = 0.0

    for t in range(n):
        e = 0.5 * z[t]  # split log-odds between the two states
        from_a = score_a - lam
        from_h = score_h - lam
        stay_h = score_h + e
        stay_a = score_a - e
        back_h[t] = from_a > stay_h
        back_a[t] = from_h > stay_a
        score_h = max(from_a, stay_h)
        score_a = max(from_h, stay_a)

    state = HOME if score_h >= score_a else AWAY
    out = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = state
        back = back_h if state == HOME else back_a
        if back[t]:
            state = AWAY if state == HOME else HOME
    return out


def run(lam=20.0, alpha=0.25, cap=2.0, name="level0_db", use_deadball=True):
    players, ball, team = load_cache()
    z = evidence(players, ball, team, alpha=alpha, cap=cap)
    codes = decode(z, lam=lam)
    labels = np.where(codes == HOME, "H", "A")
    if use_deadball:
        dead = detect_deadball(players, ball)
        labels = np.where(dead, "D", labels)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = B.write_submission(os.path.join(OUT_DIR, f"{name}.csv"), labels)
    counts = {lab: int(np.sum(labels == lab)) for lab in ("H", "A", "D")}
    print(f"{name}: lam={lam} deadball={use_deadball}")
    print(f"  H={counts['H']} A={counts['A']} D={counts['D']}")
    print(f"  wrote {path}")
    return path, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lam", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--cap", type=float, default=2.0)
    parser.add_argument("--name", default="level0")
    args = parser.parse_args()
    run(lam=args.lam, alpha=args.alpha, cap=args.cap, name=args.name)


from mtah.deadball import detect as detect_deadball


def run(lam=20.0, alpha=0.25, cap=2.0, name="level0_db", use_deadball=True):
    players, ball, team = load_cache()
    z = evidence(players, ball, team, alpha=alpha, cap=cap)
    codes = decode(z, lam=lam)
    labels = np.where(codes == HOME, "H", "A")
    if use_deadball:
        dead = detect_deadball(players, ball)
        labels = np.where(dead, "D", labels)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = B.write_submission(os.path.join(OUT_DIR, f"{name}.csv"), labels)
    counts = {lab: int(np.sum(labels == lab)) for lab in ("H", "A", "D")}
    print(f"{name}: lam={lam} deadball={use_deadball}")
    print(f"  H={counts['H']} A={counts['A']} D={counts['D']}")
    print(f"  wrote {path}")
    return path, labels
