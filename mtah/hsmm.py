#!/usr/bin/env python3
"""mtah/hsmm.py -- step 5: duration-aware decoder over {H, A, D} and the T1 tier.

Model (pre-registered, fixed before any T1 output was seen):
  Each label is a chain of substates, so its duration has an explicit prior:
    live (H or A): negative binomial, shape 2, mean 10 s (minimum 2 frames)
    dead (D)     : 48 forced frames + negative binomial shape 2, mean 18 s
                   in total (minimum 50 frames = 2 s)
  On leaving a live label: P(-> D) = 0.3, P(-> other team) = 0.7.
  On leaving D: H or A with probability 0.5 each. Initial: H, A, D 1/3 each.
  Emissions per frame (log-likelihood, up to a constant):
    H: +L/2     A: -L/2     D: Ld + log((e^{L/2} + e^{-L/2}) / 2)
  L  = possession evidence / tau   (step 2, llr_scaled)
  Ld = dead evidence / tau_d       (step 4; rules-only after its kill)
  In D the ball's proximity says nothing about the team, hence the mixture.
  Graded frames before a half's kick-off are forced to D (step 2 play mask).
  Each graded block is decoded on its own; D is written outside the blocks.
Decoding: Viterbi (MAP segmentation), then every boundary is moved to the
  split that maximises the summed forward-backward log-posteriors of the two
  adjacent labels (ties -> midpoint of the tied positions; D keeps >= 2 s).
  Reason (found on synthetic data before any real T1 output): inside long
  ball-free gaps every switch point has equal Viterbi cost, so Viterbi's
  tie-break depends on the direction of time; the label chain is reversible
  (detailed balance holds for the transition matrix), so posterior-based
  placement with midpoint ties is time-symmetric.
Implementation: vectorised numpy per frame over ~54 states; no numba needed.

T1 = step-1 ball track (mode as decided by its run) + step-2 evidence +
  step-4 dead evidence (mode as decided by its run) + this decoder. The T1
  pipeline recomputes everything from (players, ball) so step 3's dropout and
  jitter tests are end-to-end; parity with the cached step 1/2/4 arrays is
  checked on the unperturbed input.

CLI (repo root):
  python -m mtah.hsmm smoke
  python -m mtah.hsmm t1          # writes out/T1.csv, validates, applies the ship rule vs T0
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from mtah.frames import BLOCKS, D, FPS, N_FRAMES, OUT_DIR, half_index

LIVE_MEAN_S, LIVE_SHAPE = 10.0, 2
DEAD_MEAN_S, DEAD_SHAPE, DEAD_MIN_S = 18.0, 2, 2.0
P_LIVE_TO_DEAD = 0.3
P_DEAD_TO_HOME = 0.5
NEG = -1e30


# --------------------------------------------------------------------------- model
class Chain:
    """State layout: [H substates][A substates][D substates]."""

    def __init__(
        self,
        live_mean_s=LIVE_MEAN_S,
        live_shape=LIVE_SHAPE,
        dead_mean_s=DEAD_MEAN_S,
        dead_shape=DEAD_SHAPE,
        dead_min_s=DEAD_MIN_S,
        p_live_dead=P_LIVE_TO_DEAD,
        p_dead_home=P_DEAD_TO_HOME,
    ):
        mu_l = live_mean_s * FPS
        s_live = 1.0 - live_shape / mu_l
        m_fixed = int(round(dead_min_s * FPS)) - dead_shape
        mu_tail = dead_mean_s * FPS - m_fixed
        s_dead = 1.0 - dead_shape / mu_tail
        stay = []
        lab = []
        for l in (0, 1):
            stay += [s_live] * live_shape
            lab += [l] * live_shape
        stay += [0.0] * m_fixed + [s_dead] * dead_shape
        lab += [2] * (m_fixed + dead_shape)
        self.stay = np.array(stay)
        self.lab = np.array(lab, np.int64)
        self.S = len(stay)
        self.first = np.array([np.flatnonzero(self.lab == l)[0] for l in range(3)])
        self.last = np.array([np.flatnonzero(self.lab == l)[-1] for l in range(3)])
        self.nonlast = np.ones(self.S, bool)
        self.nonlast[self.last] = False
        P = np.array(
            [
                [0.0, 1 - p_live_dead, p_live_dead],
                [1 - p_live_dead, 0.0, p_live_dead],
                [p_dead_home, 1 - p_dead_home, 0.0],
            ]
        )
        self.P = P
        with np.errstate(divide="ignore"):
            self.logstay = np.log(self.stay)
            self.logadv = np.log(1.0 - self.stay)
            self.logP = np.log(P)
        init = np.zeros(self.S)
        init[self.first] = 1.0 / 3.0
        with np.errstate(divide="ignore"):
            self.loginit = np.log(init)
        self.params = dict(
            live_mean_s=live_mean_s,
            live_shape=live_shape,
            dead_mean_s=dead_mean_s,
            dead_shape=dead_shape,
            dead_min_s=dead_min_s,
            p_live_dead=p_live_dead,
            p_dead_home=p_dead_home,
            states=self.S,
        )


def viterbi(e: np.ndarray, ch: Chain) -> np.ndarray:
    """e (T,3) log-emissions -> labels u8 (T,) in {0,1,2}."""
    T, S = len(e), ch.S
    bp = np.empty((T, S), np.int16)
    delta = ch.loginit + e[0, ch.lab]
    idx = np.arange(S)
    first, last = ch.first, ch.last
    nl = ch.nonlast[:-1]
    for t in range(1, T):
        new = delta + ch.logstay
        arg = idx.copy()
        adv = np.full(S, NEG)
        adv[1:] = np.where(nl, delta[:-1] + ch.logadv[:-1], NEG)
        better = adv > new
        new = np.where(better, adv, new)
        arg = np.where(better, idx - 1, arg)
        ex = delta[last] + ch.logadv[last]
        M = ex[:, None] + ch.logP
        bi = M.argmax(0)
        bv = M[bi, np.arange(3)]
        for l in range(3):
            if bv[l] > new[first[l]]:
                new[first[l]] = bv[l]
                arg[first[l]] = last[bi[l]]
        bp[t] = arg
        delta = new + e[t, ch.lab]
    s = int(np.argmax(delta))
    path = np.empty(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = bp[t, s]
    return ch.lab[path].astype(np.uint8)


def posteriors(e: np.ndarray, ch: Chain) -> np.ndarray:
    """Scaled forward-backward; returns label posteriors (T,3) float32."""
    T, S = len(e), ch.S
    lab, first, last = ch.lab, ch.first, ch.last
    stay, adv, P = ch.stay, 1.0 - ch.stay, ch.P
    nl = ch.nonlast
    b = np.exp(e - e.max(1, keepdims=True))[:, lab]
    alpha = np.empty((T, S))
    c = np.empty(T)
    a = np.exp(ch.loginit) * b[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        p = alpha[t - 1]
        pr = p * stay
        pr[1:] += np.where(nl[:-1], p[:-1] * adv[:-1], 0.0)
        ex = p[last] * adv[last]
        pr[first] += ex @ P
        a = pr * b[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.ones(S)
    gam = np.empty((T, 3))
    g = alpha[T - 1] * beta
    gam[T - 1] = np.bincount(lab, g, 3) / g.sum()
    for t in range(T - 2, -1, -1):
        bb = b[t + 1] * beta
        nb = stay * bb
        nb[:-1] += np.where(nl[:-1], adv[:-1] * bb[1:], 0.0)
        nb[last] += adv[last] * (P @ bb[first])
        beta = nb / c[t + 1]
        g = alpha[t] * beta
        gam[t] = np.bincount(lab, g, 3) / g.sum()
    return gam.astype(np.float32)


def emissions(L, Ld, play) -> np.ndarray:
    """(n,3) log-emissions; graded pre-kick-off frames forced to D."""
    n = len(L)
    L = L.astype(np.float64)
    e = np.empty((n, 3))
    e[:, 0] = L / 2
    e[:, 1] = -L / 2
    e[:, 2] = Ld.astype(np.float64) + np.logaddexp(L / 2, -L / 2) - np.log(2.0)
    pre = (half_index(n) >= 0) & ~play
    e[pre, 0] = NEG / 1e10
    e[pre, 1] = NEG / 1e10
    return e


def refine(path, post, min_len=(1, 1, int(round(DEAD_MIN_S * FPS)))):
    """Move each boundary to argmax_c [sum_{t<c} log P(X) + sum_{t>=c} log P(Y)]
    within the two adjacent segments (left to right), ties -> midpoint."""
    lab = path.copy()
    st, en, va = _segments(lab)
    if len(st) < 2:
        return lab, 0, 0.0
    lp = np.log(np.clip(post.astype(np.float64), 1e-12, 1.0))
    moved, shift = 0, 0.0
    start_k = st[0]
    for k in range(len(st) - 1):
        X, Y = int(va[k]), int(va[k + 1])
        lo = start_k + min_len[X]
        hi = en[k + 1] - min_len[Y]
        b0 = st[k + 1]
        if lo <= hi:
            g = np.cumsum(lp[start_k:hi, X] - lp[start_k:hi, Y])  # g[c-1-start_k]
            cand = np.arange(lo, hi + 1)
            val = g[cand - 1 - start_k]
            best = val.max()
            ties = cand[val >= best - 1e-9]
            b = int(
                ties[(len(ties) - 1) // 2]
                if len(ties) % 2
                else (ties[len(ties) // 2 - 1] + ties[len(ties) // 2]) // 2
            )
            if b != b0:
                lab[min(b, b0) : max(b, b0)] = X if b > b0 else Y
                moved += 1
                shift += abs(b - b0)
            start_k = b
        else:
            start_k = b0
    return lab, moved, shift / max(moved, 1)


def _segments(lab):
    change = np.flatnonzero(np.diff(lab)) + 1
    st = np.r_[0, change]
    en = np.r_[change, len(lab)]
    return st, en, lab[st]


def decode(e, ch=None, refine_boundaries=True, info=None) -> np.ndarray:
    ch = ch or Chain()
    lab = np.full(len(e), D, np.uint8)
    for a, b in BLOCKS:
        path = viterbi(e[a : b + 1], ch)
        if refine_boundaries:
            post = posteriors(e[a : b + 1], ch)
            path, moved, mean_shift = refine(path, post)
            if info is not None:
                info.setdefault("moved", []).append(moved)
                info.setdefault("mean_shift_frames", []).append(mean_shift)
        lab[a : b + 1] = path
    return lab


def decode_posteriors(e, ch=None) -> np.ndarray:
    ch = ch or Chain()
    post = np.zeros((len(e), 3), np.float32)
    post[:, 2] = 1.0
    for a, b in BLOCKS:
        post[a : b + 1] = posteriors(e[a : b + 1], ch)
    return post


# --------------------------------------------------------------------------- T1 assembly
def dead_rules_llr(bf, ev, tags, kick_events):
    """Rules-only dead evidence, identical to deadball2 after its kill."""
    from mtah import deadball2 as DB

    g = half_index(len(ev["play"])) >= 0
    play = ev["play"].astype(bool)
    _, _, an = DB.make_anchors(
        bf["pos"],
        bf["obs"].astype(bool),
        bf["speed"],
        ev["pseudo"],
        play,
        kick_events,
        tags,
    )
    sp = np.nan_to_num(bf["speed"], nan=0.0)
    near = np.fmin(ev["d_home"], ev["d_away"])
    with np.errstate(invalid="ignore"):
        control = (
            an["inside"]
            & (near < DB.CONTROL_DIST)
            & (sp > DB.CONTROL_SPEED)
            & (sp <= DB.FAST)
        )
    rules = (
        np.where(an["oob"], DB.RULE_OOB, 0.0)
        + np.where(an["static_any"], DB.RULE_PLACED, 0.0)
        + np.where(an["inside"] & (sp > DB.FAST), DB.RULE_FAST, 0.0)
        + np.where(control, DB.RULE_CONTROL, 0.0)
    )
    d = np.clip(rules, -DB.CLIP, DB.CLIP)
    d = np.where(g & ~play, DB.RULE_PREKICK, d)
    return np.where(g, d, 0.0)


def ball_track(ball, half, mode, q):
    """Step-1 track in the same dtypes as cache/ballfilter.npz (float32), so the
    downstream evidence is bit-identical to the cached run."""
    from mtah import ballfilter as BF

    if mode == "kalman":
        out = BF.pipeline(ball, half, q)
    else:
        r = BF.estimate_r(ball, BF.visible_onpitch(ball))
        kept, _, _, _ = BF.gate(ball, half, BF.d0_for(r))
        out = BF.linear_fallback(ball, kept)
    for k in ("pos", "vel", "speed", "sigma"):
        out[k] = out[k].astype(np.float32)
    return out


class T1:
    """Holds the fixed (unperturbed) context; __call__(players, ball) -> labels."""

    def __init__(self, team, starts, tags, kick_events, bf_mode, bf_q, dead_killed):
        if not dead_killed:
            raise NotImplementedError(
                "deadball2 model kept: wire deadball2.compute here before use"
            )
        self.team, self.starts, self.tags, self.kick = team, starts, tags, kick_events
        self.bf_mode, self.bf_q = bf_mode, bf_q
        self.ch = Chain()
        self.last = {}

    def emissions(self, players, ball):
        from mtah.evidence import act_time, compute

        half = half_index(len(ball))
        bf = ball_track(ball, half, self.bf_mode, self.bf_q)
        cache = dict(players=players, ball=ball, team=self.team)
        ev, _ = compute(cache, bf, self.starts, verbose=False)
        dl = dead_rules_llr(bf, ev, self.tags, self.kick)
        g = half >= 0
        tau_d = act_time(dl, g & ev["play"], smooth=375)["tau"]
        e = emissions(ev["llr_scaled"], dl / tau_d, ev["play"])
        self.last = dict(bf=bf, ev=ev, dead_llr=dl, tau_d=tau_d)
        return e

    def __call__(self, players, ball):
        return decode(self.emissions(players, ball), self.ch)


def parity(t1, ref_bf, ref_ev, ref_db) -> dict:
    """Recomputed arrays (unperturbed) vs cached step-1/2/4 outputs."""
    lb, le = t1.last["bf"], t1.last["ev"]
    r = dict(
        ball_pos=bool(
            np.array_equal(
                np.nan_to_num(lb["pos"]).astype(np.float32),
                np.nan_to_num(ref_bf["pos"]).astype(np.float32),
            )
        ),
        evidence_llr=bool(np.array_equal(le["llr"], ref_ev["llr"])),
        pseudo=bool(np.array_equal(le["pseudo"], ref_ev["pseudo"])),
        dead_llr=bool(
            np.allclose(
                t1.last["dead_llr"].astype(np.float32), ref_db["dead_llr"], atol=1e-6
            )
        ),
    )
    return r


def run_t1(seeds=(0,)) -> int:
    from mtah import ballfilter, deadball2, evidence
    from mtah import validate as V
    from mtah.frames import (
        attack_direction,
        check_games,
        default_tags_path,
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
    dbj = json.loads(deadball2.OUT_PATH.with_suffix(".json").read_text())
    tags = load_tag_restarts(default_tags_path(), verbose=False)
    signs = attack_direction(
        cache["players"], cache["team"], half_index(), verbose=False
    )
    kick = kickoff_events(
        cache["players"], cache["team"], cache["ball"], signs, verbose=False
    ).tolist()
    bf_mode = bfj.get("mode", "linear_fallback")
    bf_q = float(bfj["diag"]["q"])
    starts = evj["play_starts"]
    print(
        f"[T1] ball track {bf_mode} (q {bf_q}); play starts {starts}; dead model killed "
        f"{dbj['model_killed']}; chain {Chain().params}"
    )
    t1 = T1(cache["team"], starts, tags, kick, bf_mode, bf_q, dbj["model_killed"])
    e = t1.emissions(cache["players"], cache["ball"])
    par = parity(t1, ballfilter.load(), evidence.load(), deadball2.load())
    print(f"[T1] parity with cached steps 1/2/4: {par}")
    if not all(par.values()):
        print(
            "[T1] PARITY FAIL: the T1 pipeline does not reproduce the cached arrays; stopping"
        )
        return 1
    td = time.time()
    rinfo = {}
    lab = decode(e, t1.ch, info=rinfo)
    dec_s = time.time() - td
    print(
        f"[T1] boundary refinement: moved {rinfo.get('moved')} boundaries per block, "
        f"mean shift {[round(x, 1) for x in rinfo.get('mean_shift_frames', [])]} frames"
    )
    post = decode_posteriors(e, t1.ch)
    g = half_index() >= 0
    conf = post.max(1)[g]
    print(
        f"[T1] decode {dec_s:.1f}s; tau_d {t1.last['tau_d']:.1f}; mean max-posterior {conf.mean():.4f}; "
        f"share < 0.6: {(conf < 0.6).mean():.4f}"
    )
    sha = write_csv(lab, OUT_DIR / "T1.csv")
    print(f"[T1] wrote {OUT_DIR / 'T1.csv'} sha256 {sha}")
    ev = evidence.load()
    res, _ = V.evaluate_tier(
        "T1",
        t1,
        cache["players"],
        cache["ball"],
        ev["play"].astype(bool),
        ev["pseudo"],
        lambda em: decode(em, t1.ch),
        e,
        seeds=tuple(seeds),
    )
    t0lab = read_csv(OUT_DIR / "T0.csv")
    V.compare(t0lab, lab, "T0", "T1")
    t0res_path = OUT_DIR / "validate_T0.json"
    if t0res_path.is_file():
        t0res = json.loads(t0res_path.read_text())
        chosen = V.ship_rule([t0res, res])
    else:
        chosen = "T0"
        print(
            "[ship] out/validate_T0.json missing: run python -m mtah.validate t0; defaulting to T0"
        )
    (OUT_DIR / "T1.diag.json").write_text(
        json.dumps(
            dict(
                sha256=sha,
                chain=t1.ch.params,
                parity=par,
                tau_d=t1.last["tau_d"],
                decode_seconds=dec_s,
                mean_max_posterior=float(conf.mean()),
                stats=label_stats(lab, verbose=False),
                ship=chosen,
                seconds=time.time() - t0,
            ),
            indent=2,
            default=float,
        )
    )
    print(f"[T1] done in {time.time() - t0:.1f}s; ship rule -> {chosen}")
    return 0


# --------------------------------------------------------------------------- smoke
def synthetic_sequence(T=40_000, seed=0, miss=0.4):
    """Labels with the prior's duration laws; noisy, partly missing emissions."""
    rng = np.random.default_rng(seed)
    lab = np.empty(T, np.uint8)
    t, cur = 0, int(rng.integers(2))
    while t < T:
        if cur == 2:
            d = 50 + int(rng.negative_binomial(2, 2 / (DEAD_MEAN_S * FPS - 48)))
        else:
            d = 2 + int(rng.negative_binomial(2, 2 / (LIVE_MEAN_S * FPS)))
        lab[t : t + d] = cur
        t += d
        if cur == 2:
            cur = int(rng.integers(2))
        else:
            cur = 2 if rng.random() < P_LIVE_TO_DEAD else 1 - cur
    L = np.where(lab == 0, 0.5, np.where(lab == 1, -0.5, 0.0)) + rng.normal(0, 0.6, T)
    L[rng.random(T) < miss] = 0.0
    Ld = np.where(lab == 2, 0.15, -0.08) + rng.normal(0, 0.3, T)
    return lab, L, Ld


def smoke() -> bool:
    from mtah.frames import possessions, spell_f1

    t0 = time.time()
    ch = Chain()
    lab, L, Ld = synthetic_sequence()
    e = np.c_[L / 2, -L / 2, Ld + np.logaddexp(L / 2, -L / 2) - np.log(2)]
    ts = time.time()
    pred = viterbi(e, ch)
    vs = time.time() - ts
    post = posteriors(e, ch)
    pr_e = np.c_[np.zeros(len(e)), np.zeros(len(e)), np.zeros(len(e))]
    acc = float((pred == lab).mean())
    blocks = ((0, len(lab) - 1),)
    sp = spell_f1(lab, pred, blocks)
    n_true = sum(len(v) for v in possessions(lab, blocks).values())
    n_pred = sum(len(v) for v in possessions(pred, blocks).values())
    rev = viterbi(e[::-1].copy(), ch)[::-1]
    rev_agree = float((rev == pred).mean())
    # long ball-free gaps: Viterbi alone is direction-dependent, refinement is not
    lab2, L2, Ld2 = synthetic_sequence(seed=1, miss=0.0)
    rng = np.random.default_rng(7)
    for s0 in rng.choice(len(L2) - 1000, 40, replace=False):
        L2[s0 : s0 + int(rng.integers(250, 1000))] = 0.0
    e2 = np.c_[L2 / 2, -L2 / 2, Ld2 + np.logaddexp(L2 / 2, -L2 / 2) - np.log(2)]
    v_f = viterbi(e2, ch)
    v_b = viterbi(e2[::-1].copy(), ch)[::-1]
    p_f = posteriors(e2, ch)
    p_b = posteriors(e2[::-1].copy(), ch)[::-1]
    r_f, _, _ = refine(v_f, p_f)
    r_b, _, _ = refine(v_b[::-1].copy(), p_b[::-1].copy())
    r_b = r_b[::-1]
    gap_rev_vit = float((v_f == v_b).mean())
    gap_rev_ref = float((r_f == r_b).mean())
    acc_vit = float((v_f == lab2).mean())
    acc_ref = float((r_f == lab2).mean())
    print(
        f"[smoke] long-gap reversal: viterbi {gap_rev_vit:.4f} -> refined {gap_rev_ref:.4f}; "
        f"accuracy viterbi {acc_vit:.4f} -> refined {acc_ref:.4f}"
    )
    psum = bool(np.allclose(post.sum(1), 1.0, atol=1e-4))
    map_agree = float((post.argmax(1) == pred).mean())
    flat = viterbi(pr_e, ch)
    dmin = (
        min((e_ - s_) for s_, e_ in zip(*_runs_of(pred, 2)))
        if (pred == 2).any()
        else 99
    )
    print(
        f"[smoke] states {ch.S}; viterbi {vs:.2f}s for {len(e)} frames ({vs / len(e) * 1e6:.1f} us/frame)"
    )
    print(
        f"[smoke] accuracy {acc:.4f}; spellF1 {sp:.4f}; possessions true {n_true} pred {n_pred}; "
        f"reversal {rev_agree:.4f}; posterior argmax == viterbi {map_agree:.4f}; min D run {dmin}"
    )
    checks = [
        ("frame accuracy >= 0.85", acc >= 0.85),
        ("possession count within 25% of truth", abs(n_pred - n_true) <= 0.25 * n_true),
        ("reversal agreement >= 0.97", rev_agree >= 0.97),
        ("posteriors sum to 1", psum),
        ("posterior argmax agrees with Viterbi >= 0.95", map_agree >= 0.95),
        ("D runs respect the 2-s minimum", dmin >= 50),
        ("no evidence -> one label, no switching", len(np.unique(flat)) == 1),
        ("throughput <= 60 us/frame", vs / len(e) <= 60e-6),
        ("long-gap reversal after refinement >= 0.99", gap_rev_ref >= 0.99),
        ("refinement does not reduce accuracy by > 0.005", acc_ref >= acc_vit - 0.005),
    ]
    ok = True
    for k, v in checks:
        ok &= bool(v)
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"[smoke] {time.time() - t0:.1f}s -> {'PASS' if ok else 'FAIL'}")
    return ok


def _runs_of(lab, val):
    m = np.r_[False, lab == val, False]
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("t1")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        return 0 if smoke() else 1
    return run_t1(tuple(a.seeds))


if __name__ == "__main__":
    sys.exit(main())
