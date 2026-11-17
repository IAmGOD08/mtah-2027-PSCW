#!/usr/bin/env python3
"""mtah/restarts.py -- step 6: stoppages from restart tags and the ball; tier T0R.

Why: T1 (step 5) under-segmented (126 possessions) and the learned dead-ball
model was killed (step 4), so T0 has no D at all and runs straight through
every stoppage. Restart tags are well timed: their counts match the task
statement exactly and kick-off tags lead the formation break by a consistent
85.5 frames (MAD 7.5, both halves). A tagged restart is a stoppage that ends
at a known moment, i.e. a possession boundary T0 is missing.

Stoppage windows [S, R) on graded, in-play frames (all label-free):
  restart tags (throw-in, corner, goal kick, free kick, kick-off after goal):
    R = tag frame - measured tag offset (frames.kickoff_alignment; -85.5 -> +86);
        kick-offs after goals snap to a detected kick-off formation within 10 s.
    S = 1 + last frame of live ball evidence in [R - max, R - 2 s], where live
        evidence = observed ball inside the pitch that is either kicked (> 8 m/s)
        or under control (within 1.5 m of a player, > 2 m/s) -- the step-4 rules;
        if the ball gives no live evidence there, S = R - typical.
    (typical, max) seconds, football norms fixed before use: throw-in (8, 15),
        corner (20, 30), goal kick (18, 25), free kick (25, 40), kick-off after
        goal (45, 60). Every window lasts >= 2 s.
  observed out of bounds (untagged stoppages): each run of observed ball > 0.5 m
    outside the lines starts a window that ends at the next live ball evidence,
    capped at 40 s, at least 2 s.
  pre-kick-off: graded frames before each half's kick-off (step 2).
Labels: D inside the windows; every live stretch between windows is decoded
on its own with the T0 decoder (level0 evidence, two-state Viterbi, lam 20), so
the team after a stoppage is decided by the evidence after it.

Kill criteria (declared before any T0R output): the harness (step 3, rev 1)
bands, pseudo-label agreement >= 0.90, reversal >= 0.97, runtime <= 120 s; and
here: D on pseudo-label frames <= 5% (a stoppage window should rarely cover a
ball at a player's feet). Ship rule over [T0, T1, T0R] as in step 3.

CLI (repo root):
  python -m mtah.restarts smoke
  python -m mtah.restarts tier     # out/T0R.csv, out/validate_T0R.json, out/SHIP.json, out/submission.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import numpy as np

from mtah.frames import (
    BLOCKS,
    D,
    FPS,
    N_FRAMES,
    OUT_DIR,
    PITCH_X,
    PITCH_Y,
    half_index,
    runs,
)

PRIORS_S = dict(
    throw_in=(8, 15),
    corner=(20, 30),
    goal_kick=(18, 25),
    free_kick=(25, 40),
    kick_off_goal=(45, 60),
)
OOB_M, INSIDE_M, FAST, CONTROL_DIST, CONTROL_SPEED = 0.5, 1.0, 8.0, 1.5, 2.0
OOB_MAX_S = 40.0
MIN_D = 50
SNAP = 250
D_ON_PSEUDO_MAX = 0.05


# --------------------------------------------------------------------------- ball cues
def ball_cues(bf, ev):
    pos, obs = bf["pos"], bf["obs"].astype(bool)
    sp = np.nan_to_num(bf["speed"], nan=0.0)
    x, y = pos[:, 0], pos[:, 1]
    with np.errstate(invalid="ignore"):
        inside = (
            obs
            & (x > INSIDE_M)
            & (x < PITCH_X - INSIDE_M)
            & (y > INSIDE_M)
            & (y < PITCH_Y - INSIDE_M)
        )
        oob = obs & (
            (x < -OOB_M) | (x > PITCH_X + OOB_M) | (y < -OOB_M) | (y > PITCH_Y + OOB_M)
        )
        near = np.fmin(ev["d_home"], ev["d_away"])
        live = inside & ((sp > FAST) | ((near < CONTROL_DIST) & (sp > CONTROL_SPEED)))
    return live, oob


def stoppage_windows(
    tags, tag_offset, kick_events, live, oob, play
) -> tuple[np.ndarray, list]:
    n = len(live)
    g = half_index(n) >= 0
    ok = g & play
    live_idx = np.flatnonzero(live)
    kick = np.asarray(kick_events, np.int64)
    wins = []
    for f, kind in tags:
        if kind not in PRIORS_S:
            continue
        typ, mx = (int(round(v * FPS)) for v in PRIORS_S[kind])
        R = int(round(f - tag_offset))
        if kind == "kick_off_goal" and len(kick):
            j = int(np.abs(kick - R).argmin())
            if abs(kick[j] - R) <= SNAP:
                R = int(kick[j])
        lo, hi = R - mx, R - MIN_D
        k = live_idx[(live_idx >= lo) & (live_idx < hi)]
        if len(k):
            S, src = int(k[-1]) + 1, "ball"
        else:
            S, src = R - typ, "prior"
        S = min(max(S, lo), R - MIN_D)
        wins.append((S, R, kind, src))
    s, e, v = runs(oob.astype(np.uint8))
    for a, b in zip(s[v == 1], e[v == 1]):
        nxt = live_idx[live_idx >= b]
        end = int(nxt[0]) if len(nxt) else a + int(OOB_MAX_S * FPS)
        end = min(end, a + int(OOB_MAX_S * FPS))
        end = max(end, a + MIN_D)
        wins.append((int(a), int(end), "out_of_bounds", "ball"))
    mask = np.zeros(n, bool)
    for S, R, _, _ in wins:
        mask[max(S, 0) : min(R, n)] = True
    mask &= ok
    mask |= g & ~play
    return mask, wins


def t0r_labels(z, dmask, lam=None) -> np.ndarray:
    from mtah.frames import T0_PARAMS, viterbi2

    lam = T0_PARAMS["lam"] if lam is None else lam
    n = len(z)
    lab = np.full(n, D, np.uint8)
    for a, b in BLOCKS:
        seg = dmask[a : b + 1]
        s, e, v = runs(seg.astype(np.uint8))
        for ss, ee, vv in zip(s, e, v):
            if vv == 0:
                lab[a + ss : a + ee] = viterbi2(z[a + ss : a + ee], lam)
    return lab


class T0R:
    """Fixed context; __call__(players, ball) -> labels (end-to-end for the harness)."""

    def __init__(self, team, starts, tags, tag_offset, kick_events, bf_mode, bf_q):
        self.team, self.starts, self.tags = team, starts, tags
        self.offset, self.kick, self.bf_mode, self.bf_q = (
            tag_offset,
            kick_events,
            bf_mode,
            bf_q,
        )
        self.last = {}

    def emissions(self, players, ball):
        from mtah.evidence import compute
        from mtah.frames import soft_evidence
        from mtah.hsmm import ball_track

        half = half_index(len(ball))
        bf = ball_track(ball, half, self.bf_mode, self.bf_q)
        ev, _ = compute(
            dict(players=players, ball=ball, team=self.team),
            bf,
            self.starts,
            verbose=False,
        )
        live, oob = ball_cues(bf, ev)
        dmask, wins = stoppage_windows(
            self.tags, self.offset, self.kick, live, oob, ev["play"].astype(bool)
        )
        z = soft_evidence(players, ball, self.team)
        self.last = dict(wins=wins, dmask=dmask)
        return np.c_[z, dmask.astype(np.float32)]

    def __call__(self, players, ball):
        em = self.emissions(players, ball)
        return decode_fn(em)


def decode_fn(em):
    return t0r_labels(em[:, 0].astype(np.float32), em[:, 1] > 0.5)


def window_summary(wins, verbose=True) -> dict:
    kinds = {}
    for S, R, k, src in wins:
        d = kinds.setdefault(k, dict(n=0, from_ball=0, seconds=[]))
        d["n"] += 1
        d["from_ball"] += src == "ball"
        d["seconds"].append((R - S) / FPS)
    out = {
        k: dict(
            n=v["n"],
            onset_from_ball=v["from_ball"],
            median_s=float(np.median(v["seconds"])),
        )
        for k, v in kinds.items()
    }
    if verbose:
        print(
            "[T0R] stoppage windows by source (count, onset from ball evidence, median length):"
        )
        for k, v in out.items():
            print(
                f"  {k:15s} {v['n']:4d}  {v['onset_from_ball']:4d}  {v['median_s']:6.1f} s"
            )
    return out


# --------------------------------------------------------------------------- tier
def run_tier(seeds=(0,)) -> int:
    from mtah import ballfilter, evidence
    from mtah import validate as V
    from mtah.frames import (
        attack_direction,
        check_games,
        default_tags_path,
        kickoff_alignment,
        kickoff_events,
        label_stats,
        load_tag_restarts,
        load_tracking,
        read_csv,
        write_csv,
    )

    t0 = time.time()
    check_games()
    cache = load_tracking()
    bfj = json.loads(ballfilter.OUT_PATH.with_suffix(".json").read_text())
    evj = json.loads(evidence.OUT_PATH.with_suffix(".json").read_text())
    tags = load_tag_restarts(default_tags_path(), verbose=False)
    signs = attack_direction(
        cache["players"], cache["team"], half_index(), verbose=False
    )
    kick = kickoff_events(
        cache["players"], cache["team"], cache["ball"], signs, verbose=False
    )
    al = kickoff_alignment(
        [(f, k) for f, k in tags if k.startswith("kick_off")], kick, verbose=False
    )
    if not al["pass"]:
        print("[T0R] tag alignment check failed: refusing to place stoppages from tags")
        return 1
    off = float(al["semantic_offset_frames"])
    tier = T0R(
        cache["team"],
        evj["play_starts"],
        tags,
        off,
        kick.tolist(),
        bfj.get("mode", "linear_fallback"),
        float(bfj["diag"]["q"]),
    )
    em = tier.emissions(cache["players"], cache["ball"])
    print(f"[T0R] tag offset {off} frames; play starts {evj['play_starts']}")
    ws = window_summary(tier.last["wins"])
    lab = decode_fn(em)
    sha = write_csv(lab, OUT_DIR / "T0R.csv")
    print(f"[T0R] wrote {OUT_DIR / 'T0R.csv'} sha256 {sha}")
    ev = evidence.load()
    res, _ = V.evaluate_tier(
        "T0R",
        tier,
        cache["players"],
        cache["ball"],
        ev["play"].astype(bool),
        ev["pseudo"],
        decode_fn,
        em,
        seeds=tuple(seeds),
        save=False,
    )
    d_on_pseudo = res["report"]["pseudo_frames_labelled_D"]
    res["checks"][f"D on pseudo frames <= {D_ON_PSEUDO_MAX}"] = (
        d_on_pseudo <= D_ON_PSEUDO_MAX
    )
    res["kill_pass"] = all(res["checks"].values())
    print(
        f"  {'PASS' if d_on_pseudo <= D_ON_PSEUDO_MAX else 'FAIL'}  D on pseudo frames {d_on_pseudo:.4f} "
        f"<= {D_ON_PSEUDO_MAX}  -> {'eligible' if res['kill_pass'] else 'NOT eligible'}"
    )
    res["windows"] = ws
    (OUT_DIR / "validate_T0R.json").write_text(json.dumps(res, indent=2, default=float))
    t0lab = read_csv(OUT_DIR / "T0.csv")
    V.compare(t0lab, lab, "T0", "T0R")
    tiers = []
    for name in ("T0", "T1"):
        p = OUT_DIR / f"validate_{name}.json"
        if p.is_file():
            tiers.append(json.loads(p.read_text()))
    if not tiers or tiers[0]["name"] != "T0":
        print(
            "[ship] out/validate_T0.json missing: run python -m mtah.validate t0 first"
        )
        return 1
    chosen = V.ship_rule(tiers + [res])
    src = OUT_DIR / f"{chosen}.csv"
    shutil.copyfile(src, OUT_DIR / "submission.csv")
    ship = dict(
        chosen=chosen,
        source=str(src),
        sha256_T0R=sha,
        candidates={
            t["name"]: dict(
                kill_pass=t["kill_pass"], dropout=t["dropout_hidden_agreement"]
            )
            for t in tiers + [res]
        },
        stats=label_stats(read_csv(src), verbose=False),
        seconds=time.time() - t0,
    )
    (OUT_DIR / "SHIP.json").write_text(json.dumps(ship, indent=2, default=float))
    print(
        f"[ship] out/submission.csv <- {src.name}; details in out/SHIP.json ({time.time() - t0:.1f}s)"
    )
    return 0


# --------------------------------------------------------------------------- smoke
def smoke() -> bool:
    from mtah.ballfilter import pipeline as bf_pipeline
    from mtah.evidence import compute
    from mtah.frames import soft_evidence, spell_f1, synthetic_match, t0_decode

    t0 = time.time()
    cache, truth, tags = synthetic_match(seed=0)
    half = half_index()
    bf = bf_pipeline(cache["ball"], half, q=100.0)
    bf = {
        k: (v.astype(np.float32) if k in ("pos", "vel", "speed", "sigma") else v)
        for k, v in bf.items()
    }
    starts = [a for a, _ in BLOCKS]
    ev, _ = compute(cache, bf, starts, verbose=False)
    live, oob = ball_cues(bf, ev)
    # synthetic tags mark the last static frame before the restart: offset 0
    dmask, wins = stoppage_windows(
        tags, 0.0, starts, live, oob, ev["play"].astype(bool)
    )
    window_summary(wins)
    z = soft_evidence(cache["players"], cache["ball"], cache["team"])
    lab = t0r_labels(z, dmask)
    base, _ = t0_decode(cache["players"], cache["team"], cache["ball"])
    g = half >= 0
    td = g & (truth == D)
    prec = float(td[dmask & g].mean()) if (dmask & g).any() else 0.0
    rec = float(dmask[td].mean())
    s_t0, s_r = spell_f1(truth, base), spell_f1(truth, lab)
    em = np.c_[z, dmask.astype(np.float32)]
    from mtah.validate import reverse_blocks

    rev = reverse_blocks(decode_fn(reverse_blocks(em)))
    rev_agree = float((rev[g] == lab[g]).mean())
    lab2 = t0r_labels(z, dmask)
    print(
        f"[smoke] D windows: precision {prec:.3f} recall {rec:.3f}; spellF1 vs synthetic truth "
        f"T0 {s_t0:.4f} -> T0R {s_r:.4f}; reversal {rev_agree:.4f}"
    )
    checks = [
        ("D window precision vs synthetic truth >= 0.8", prec >= 0.8),
        ("T0R spellF1 > T0 spellF1 on synthetic truth", s_r > s_t0),
        ("reversal agreement >= 0.97", rev_agree >= 0.97),
        ("ungraded frames are D", bool((lab[~g] == D).all())),
        ("deterministic", bool(np.array_equal(lab, lab2))),
    ]
    ok = True
    for k, v in checks:
        ok &= bool(v)
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"[smoke] {time.time() - t0:.1f}s -> {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("tier")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        return 0 if smoke() else 1
    return run_tier(tuple(a.seeds))


if __name__ == "__main__":
    sys.exit(main())
