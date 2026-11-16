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

from mtah.deadball import detect as detect_deadball

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


def decode3(z_ha, z_d, lam=20.0, lam_d=15.0):
    """Three-state Viterbi over {H, A, D}."""
    n = len(z_ha)
    NEG = -1e9
    score = np.array([0.0, 0.0, NEG])
    back = np.zeros((n, 3), dtype=np.int8)

    for t in range(n):
        e_ha = 0.5 * z_ha[t]
        e_d = z_d[t]
        new_score = np.empty(3)
        for s in range(3):
            best_val, best_prev = NEG, 0
            for p in range(3):
                cost = 0.0
                if p != s:
                    cost = lam_d if (s == 2 or p == 2) else lam
                val = score[p] - cost
                if val > best_val:
                    best_val, best_prev = val, p
            if s == 0:
                new_score[s] = best_val + e_ha
            elif s == 1:
                new_score[s] = best_val - e_ha
            else:
                new_score[s] = best_val + e_d
            back[t][s] = best_prev
        score = new_score

    state = int(np.argmax(score))
    out = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = state
        state = back[t][state]
    return out


def dead_evidence(dead_mask, strength=1.0):
    """Convert a boolean dead-ball mask into a per-frame evidence score.

    Positive = evidence for dead ball. 0 = no evidence either way.
    """
    return np.where(dead_mask, strength, 0.0).astype(np.float32)


def run(lam=20.0, lam_d=15.0, alpha=0.25, cap=2.0, name="level0_3state"):
    players, ball, team = load_cache()
    z_ha = evidence(players, ball, team, alpha=alpha, cap=cap)
    dead = detect_deadball(players, ball)
    z_d = dead_evidence(dead, strength=1.0)
    codes = decode3(z_ha, z_d, lam=lam, lam_d=lam_d)
    labels = np.array(["H", "A", "D"])[codes]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = B.write_submission(os.path.join(OUT_DIR, f"{name}.csv"), labels)
    counts = {lab: int(np.sum(labels == lab)) for lab in ("H", "A", "D")}
    print(f"{name}: lam={lam} lam_d={lam_d}")
    print(f"  H={counts['H']} A={counts['A']} D={counts['D']}")
    print(f"  wrote {path}")
    return path, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lam", type=float, default=20.0)
    parser.add_argument("--lam_d", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--cap", type=float, default=2.0)
    parser.add_argument("--name", default="level0_3state")
    args = parser.parse_args()
    run(lam=args.lam, lam_d=args.lam_d, alpha=args.alpha, cap=args.cap, name=args.name)
