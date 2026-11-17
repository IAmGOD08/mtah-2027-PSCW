#!/usr/bin/env python3
"""mtah/ballfilter.py -- step 1: ball track reconstruction.

Pipeline (label-free, deterministic):
  1. Off-pitch rejection: observations more than 10 m outside the pitch.
  2. Physics gating: consecutive kept observations implying more than
     vmax = 40 m/s plus a noise allowance d0 = max(0.5 m, 3 x sqrt(2) x r)
     are an implausible jump (r = measurement noise, estimated first). Short
     fragments (<= 5 observations) bounded by implausible jumps on both sides,
     or by a jump on one side to a long fragment (>= 10 observations) and a gap
     or end on the other, are rejected as spikes. Repeated until stable.
  3. Segmentation: kept observations are chained into segments; a segment
     breaks at gaps longer than 75 frames (3 s), at remaining implausible
     jumps, and at every change of graded block (halves decoded separately).
  4. Constant-velocity Kalman filter + RTS smoother per segment, per axis
     (white-acceleration process noise q, measurement noise r). Inside a
     segment, gaps are bridged with posterior uncertainty that grows through
     the gap. Outside segments there is no estimate (pos NaN, sigma inf).
  5. r is estimated from the data (robust second differences); q is chosen
     from a fixed grid by the label-free dropout test: hide real observed
     1-s stretches, reconstruct, minimise the mean Gaussian negative
     log-likelihood of the hidden observations (accuracy AND calibrated
     sigma, because sigma weights the evidence downstream). Pre-registered.

Assumptions:
  B1  Ball positions are metres on the 105.3 x 68 pitch, NaN = missing.
  B2  The tracker's ball is the match ball when it is on the pitch; warm-up
      balls before the second-half kick-off are NOT removed here (step 4
      masks pre-kick-off frames as D).
  B3  No height: a lofted ball projects onto the ground plane; "inflight" only
      means fast (speed > 7 m/s with sigma < 1 m), not airborne.

Outputs (cache/ballfilter.npz, all length n = 199,319):
  pos (n,2) f32 m, NaN where no estimate     vel (n,2) f32 m/s
  speed (n,) f32 m/s                         sigma (n,) f32 m (per-axis std; inf if none)
  obs (n,) bool kept observation             rejected (n,) bool (spike or off-pitch)
  inflight (n,) bool                         segment (n,) i32, -1 outside segments

CLI (repo root):
  python -m mtah.ballfilter smoke            # synthetic, known truth, no match data
  python -m mtah.ballfilter run [--q Q]      # real data -> cache/ballfilter.npz
Kill criteria (printed), all at 1-s dropout unless stated: rejected share
<= 5%; p99 filtered speed on observed frames <= 40 m/s; median error <= 3 m;
p90 error <= linear-interpolation p90 and median <= linear median + 0.15 m;
share within 2 sigma in [0.75, 0.97] (target 0.865). Any failure -> exit 1
and the linear fallback (kept ball + linear interpolation for gaps <= 12).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from mtah.frames import BLOCKS, FPS, N_FRAMES, PITCH_X, PITCH_Y, ROOT, half_index

try:  # numba is optional; the pure-Python path gives identical results
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


VMAX = 40.0  # m/s, gating speed
D0_MIN = 0.5  # m, floor of the noise allowance on a jump
MAX_SPIKE = 5  # observations
LONG_FRAG = 10  # observations
G_BRIDGE = 75  # frames; longer gaps break a segment
MARGIN = 10.0  # m outside the pitch before an observation is rejected
V_FLIGHT = 7.0  # m/s
Q_GRID = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)  # m^2/s^3
V0_VAR = 15.0**2  # (m/s)^2 initial velocity variance
DROP_LENGTHS = (12, 25, 50)
OUT_PATH = ROOT / "cache" / "ballfilter.npz"


# --------------------------------------------------------------------------- gating
def d0_for(r: float) -> float:
    return max(D0_MIN, 3.0 * np.sqrt(2.0) * r)


def gate(
    ball: np.ndarray,
    half: np.ndarray,
    d0: float,
    vmax=VMAX,
    max_spike=MAX_SPIKE,
    long_frag=LONG_FRAG,
    g_bridge=G_BRIDGE,
    margin=MARGIN,
):
    """Returns kept (n,) bool, offpitch (n,) bool, spike (n,) bool, iterations."""
    n = len(ball)
    vis = np.isfinite(ball).all(1)
    off = vis & (
        (ball[:, 0] < -margin)
        | (ball[:, 0] > PITCH_X + margin)
        | (ball[:, 1] < -margin)
        | (ball[:, 1] > PITCH_Y + margin)
    )
    kept = vis & ~off
    spike = np.zeros(n, bool)
    it = 0
    for it in range(1, 6):
        idx = np.flatnonzero(kept)
        if len(idx) < 2:
            break
        jump, chain = _pairs(ball, half, idx, vmax, d0, g_bridge)
        brk = jump | chain
        frag = np.r_[0, np.cumsum(brk)]
        size = np.bincount(frag)
        first = np.flatnonzero(np.r_[True, brk])
        last = np.r_[first[1:] - 1, len(idx) - 1]
        lj = np.r_[False, jump][first]
        rj = np.r_[jump, False][last]
        ls = np.r_[0, size[:-1]]
        rs = np.r_[size[1:], 0]
        short = size <= max_spike
        rej = short & (
            (lj & rj)
            | (lj & ~rj & (ls >= long_frag) & (size < ls))
            | (rj & ~lj & (rs >= long_frag) & (size < rs))
        )
        if not rej.any():
            break
        bad = idx[rej[frag]]
        spike[bad] = True
        kept[bad] = False
    return kept, off, spike, it


def _pairs(ball, half, idx, vmax, d0, g_bridge):
    """Flags for consecutive kept observations: implausible jump, chain break."""
    gapf = np.diff(idx)
    dist = np.linalg.norm(np.diff(ball[idx].astype(np.float64), axis=0), axis=1)
    chain = (gapf - 1 > g_bridge) | (half[idx[1:]] != half[idx[:-1]])
    jump = ~chain & (dist > vmax * gapf / FPS + d0)
    return jump, chain


def segments(ball, half, kept, d0: float, vmax=VMAX, g_bridge=G_BRIDGE):
    """List of (first_frame, last_frame) inclusive; count of remaining jumps."""
    idx = np.flatnonzero(kept)
    if len(idx) == 0:
        return [], 0
    jump, chain = _pairs(ball, half, idx, vmax, d0, g_bridge)
    brk = jump | chain
    first = np.flatnonzero(np.r_[True, brk])
    last = np.r_[first[1:] - 1, len(idx) - 1]
    return list(zip(idx[first].tolist(), idx[last].tolist())), int(jump.sum())


# --------------------------------------------------------------------------- Kalman / RTS
@njit(cache=True)
def _rts_segment(z, has, q, r2, dt, v0var):
    """Constant-velocity RTS smoother, x and y share covariance. z (T,2), has (T,)
    with has[0] True. Returns pos (T,2), vel (T,2), P11 (T,), P22 (T,)."""
    T = z.shape[0]
    pf = np.empty((T, 2))
    vf = np.empty((T, 2))
    pp = np.empty((T, 2))
    vp = np.empty((T, 2))
    a_f = np.empty(T)
    b_f = np.empty(T)
    c_f = np.empty(T)
    a_p = np.empty(T)
    b_p = np.empty(T)
    c_p = np.empty(T)
    q11 = q * dt**3 / 3.0
    q12 = q * dt**2 / 2.0
    q22 = q * dt
    a = r2
    b = 0.0
    c = v0var
    px = z[0, 0]
    py = z[0, 1]
    vx = 0.0
    vy = 0.0
    pp[0, 0] = px
    pp[0, 1] = py
    vp[0, 0] = 0.0
    vp[0, 1] = 0.0
    a_p[0] = a
    b_p[0] = b
    c_p[0] = c
    pf[0, 0] = px
    pf[0, 1] = py
    vf[0, 0] = 0.0
    vf[0, 1] = 0.0
    a_f[0] = a
    b_f[0] = b
    c_f[0] = c
    for t in range(1, T):
        px = px + dt * vx
        py = py + dt * vy
        a2 = a + 2.0 * dt * b + dt * dt * c + q11
        b2 = b + dt * c + q12
        c2 = c + q22
        pp[t, 0] = px
        pp[t, 1] = py
        vp[t, 0] = vx
        vp[t, 1] = vy
        a_p[t] = a2
        b_p[t] = b2
        c_p[t] = c2
        if has[t]:
            s = a2 + r2
            k1 = a2 / s
            k2 = b2 / s
            ix = z[t, 0] - px
            iy = z[t, 1] - py
            px = px + k1 * ix
            py = py + k1 * iy
            vx = vx + k2 * ix
            vy = vy + k2 * iy
            a = (1.0 - k1) * a2
            b = (1.0 - k1) * b2
            c = c2 - k2 * b2
        else:
            a = a2
            b = b2
            c = c2
        pf[t, 0] = px
        pf[t, 1] = py
        vf[t, 0] = vx
        vf[t, 1] = vy
        a_f[t] = a
        b_f[t] = b
        c_f[t] = c
    ps = pf.copy()
    vs = vf.copy()
    P11 = a_f.copy()
    P22 = c_f.copy()
    sa = a_f[T - 1]
    sb = b_f[T - 1]
    sc = c_f[T - 1]
    for t in range(T - 2, -1, -1):
        a = a_f[t]
        b = b_f[t]
        c = c_f[t]
        A = a_p[t + 1]
        B = b_p[t + 1]
        C = c_p[t + 1]
        det = A * C - B * B
        m11 = a + dt * b
        m12 = b
        m21 = b + dt * c
        m22 = c  # Pf F^T
        g11 = (m11 * C - m12 * B) / det
        g12 = (-m11 * B + m12 * A) / det
        g21 = (m21 * C - m22 * B) / det
        g22 = (-m21 * B + m22 * A) / det
        for ax in range(2):
            dp = ps[t + 1, ax] - pp[t + 1, ax]
            dv = vs[t + 1, ax] - vp[t + 1, ax]
            ps[t, ax] = pf[t, ax] + g11 * dp + g12 * dv
            vs[t, ax] = vf[t, ax] + g21 * dp + g22 * dv
        da = sa - A
        db = sb - B
        dc = sc - C
        # G dP G^T, dP symmetric [[da,db],[db,dc]]
        t11 = g11 * da + g12 * db
        t12 = g11 * db + g12 * dc
        t21 = g21 * da + g22 * db
        t22 = g21 * db + g22 * dc
        sa = a + t11 * g11 + t12 * g12
        sb = b + t11 * g21 + t12 * g22
        sc = c + t21 * g21 + t22 * g22
        P11[t] = sa
        P22[t] = sc
    return ps, vs, P11, P22


def visible_onpitch(ball, margin=MARGIN):
    vis = np.isfinite(ball).all(1)
    return vis & ~(
        (ball[:, 0] < -margin)
        | (ball[:, 0] > PITCH_X + margin)
        | (ball[:, 1] < -margin)
        | (ball[:, 1] > PITCH_Y + margin)
    )


def estimate_r(ball, kept) -> float:
    """Measurement noise std (m) from robust second differences of consecutive
    kept observations: Var(z[t+1] - 2 z[t] + z[t-1]) ~ 6 r^2 for slow motion."""
    k = kept[1:-1] & kept[:-2] & kept[2:]
    t = np.flatnonzero(k) + 1
    if len(t) < 100:
        return 0.2
    d2 = (ball[t + 1] - 2 * ball[t] + ball[t - 1]).astype(np.float64).ravel()
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(np.clip(1.4826 * mad / np.sqrt(6.0), 0.03, 1.0))


def smooth(ball, kept, segs, q, r) -> dict:
    n = len(ball)
    pos = np.full((n, 2), np.nan)
    vel = np.full((n, 2), np.nan)
    p11 = np.full(n, np.inf)
    seg = np.full(n, -1, np.int32)
    z = np.where(kept[:, None], ball, np.nan).astype(np.float64)
    for i, (s, e) in enumerate(segs):
        ps, vs, P11, _ = _rts_segment(
            z[s : e + 1], kept[s : e + 1], float(q), float(r * r), 1.0 / FPS, V0_VAR
        )
        pos[s : e + 1], vel[s : e + 1], p11[s : e + 1] = ps, vs, P11
        seg[s : e + 1] = i
    return dict(pos=pos, vel=vel, sigma=np.sqrt(p11), segment=seg)


def pipeline(ball, half, q, r=None) -> dict:
    if r is None:
        r = estimate_r(ball, visible_onpitch(ball))
    d0 = d0_for(r)
    kept, off, spike, iters = gate(ball, half, d0)
    segs, n_jumps = segments(ball, half, kept, d0)
    out = smooth(ball, kept, segs, q, r)
    speed = np.linalg.norm(out["vel"], axis=1)
    out.update(
        obs=kept,
        rejected=off | spike,
        offpitch=off,
        spike=spike,
        speed=speed,
        inflight=np.nan_to_num(speed) > V_FLIGHT,
        r=r,
        q=q,
        n_segments=len(segs),
        n_jumps=n_jumps,
        gate_iterations=iters,
        d0=d0,
    )
    out["inflight"] &= out["sigma"] < 1.0
    return out


def linear_fallback(ball, kept, max_gap: int = 12) -> dict:
    """Kill-criterion fallback: kept observations + linear interpolation for
    gaps <= max_gap frames; sigma 0.3 m observed, growing 0.1 m per frame."""
    n = len(ball)
    pos = np.full((n, 2), np.nan)
    sigma = np.full(n, np.inf)
    idx = np.flatnonzero(kept)
    pos[idx] = ball[idx]
    sigma[idx] = 0.3
    for i, j in zip(idx[:-1], idx[1:]):
        g = j - i - 1
        if 0 < g <= max_gap:
            w = (np.arange(1, g + 1) / (g + 1))[:, None]
            pos[i + 1 : j] = (1 - w) * ball[i] + w * ball[j]
            sigma[i + 1 : j] = 0.3 + 0.1 * np.minimum(
                np.arange(1, g + 1), np.arange(g, 0, -1)
            )
    vel = np.full((n, 2), np.nan)
    vel[1:] = np.diff(pos, axis=0) * FPS
    speed = np.linalg.norm(vel, axis=1)
    return dict(
        pos=pos,
        vel=vel,
        speed=speed,
        sigma=sigma,
        obs=kept,
        inflight=np.nan_to_num(speed) > V_FLIGHT,
        segment=np.where(np.isfinite(sigma), 0, -1),
    )


# --------------------------------------------------------------------------- dropout test
def pick_windows(kept, half, L, n_per, seed, margin=5, spacing=250):
    """Start frames of hidden windows: kept for [s-margin, s+L+margin), graded,
    pairwise >= spacing apart. Deterministic for a seed."""
    ok = kept & (half >= 0)
    cs = np.r_[0, np.cumsum(ok)]
    s = np.arange(margin, len(kept) - L - margin)
    good = s[(cs[s + L + margin] - cs[s - margin]) == L + 2 * margin]
    rng = np.random.default_rng(seed)
    chosen, taken = [], np.zeros(len(kept) + spacing, bool)
    for c in rng.permutation(good):
        if not taken[max(c - spacing, 0) : c + L + spacing].any():
            chosen.append(int(c))
            taken[c : c + L] = True
            if len(chosen) >= n_per:
                break
    return np.sort(np.array(chosen, np.int64))


def dropout_test(ball, half, q, r, lengths=DROP_LENGTHS, n_per=300, seed=0) -> dict:
    kept0, _, _, _ = gate(ball, half, d0_for(r))
    res = {}
    for L in lengths:
        starts = pick_windows(kept0, half, L, n_per, seed + L)
        if len(starts) == 0:
            res[L] = None
            continue
        hidden = np.zeros(len(ball), bool)
        for s in starts:
            hidden[s : s + L] = True
        b2 = ball.copy()
        b2[hidden] = np.nan
        out = pipeline(b2, half, q, r)
        hid = np.flatnonzero(hidden)
        est = out["pos"][hid]
        have = np.isfinite(est).all(1)
        err = np.linalg.norm(est[have] - ball[hid][have], axis=1)
        sig = np.sqrt(out["sigma"][hid][have] ** 2 + r * r)
        nll = np.log(2 * np.pi * sig**2) + err**2 / (2 * sig**2)
        lin = []
        for s in starts:
            a, b_ = ball[s - 1].astype(float), ball[s + L].astype(float)
            w = (np.arange(1, L + 1) / (L + 1))[:, None]
            lin.append(np.linalg.norm((1 - w) * a + w * b_ - ball[s : s + L], axis=1))
        lin = np.concatenate(lin)
        res[L] = dict(
            windows=int(len(starts)),
            coverage=float(have.mean()),
            median=float(np.median(err)),
            p90=float(np.percentile(err, 90)),
            linear_median=float(np.median(lin)),
            linear_p90=float(np.percentile(lin, 90)),
            within_2sigma=float((err <= 2 * sig).mean()),
            nll=float(nll.mean()),
        )
    return res


def select_q(ball, half, r, grid=Q_GRID, verbose=True) -> tuple[float, dict]:
    """Pre-registered: q minimising mean NLL of hidden 1-s (25-frame) observations."""
    table = {}
    for q in grid:
        table[q] = dropout_test(ball, half, q, r, lengths=(25,))[25]
        if verbose:
            t = table[q]
            print(
                f"  q {q:7.1f}: 1-s gap NLL {t['nll']:.3f}  median {t['median']:.3f} m  p90 {t['p90']:.3f}  "
                f"(linear {t['linear_median']:.3f}/{t['linear_p90']:.3f})  within 2 sigma {t['within_2sigma']:.3f}"
            )
    best = min(grid, key=lambda q: table[q]["nll"])
    return best, table


# --------------------------------------------------------------------------- diagnostics
def diagnostics(ball, half, out, drop, verbose=True) -> dict:
    vis = np.isfinite(ball).all(1)
    g = half >= 0
    obs = out["obs"]
    sp_obs = out["speed"][obs & np.isfinite(out["speed"])]
    gap = np.isfinite(out["sigma"]) & ~obs
    d = dict(
        r_m=out["r"],
        d0_m=out["d0"],
        q=out["q"],
        numba=HAVE_NUMBA,
        visible=int(vis.sum()),
        kept=int(obs.sum()),
        rejected_share=float(out["rejected"].sum() / max(vis.sum(), 1)),
        offpitch=int(out["offpitch"].sum()),
        spikes=int(out["spike"].sum()),
        gate_iterations=out["gate_iterations"],
        segments=out["n_segments"],
        remaining_jumps=out["n_jumps"],
        coverage_graded=float(np.isfinite(out["sigma"])[g].mean()),
        raw_visible_graded=float(vis[g].mean()),
        speed_p50=float(np.percentile(sp_obs, 50)),
        speed_p90=float(np.percentile(sp_obs, 90)),
        speed_p99=float(np.percentile(sp_obs, 99)),
        speed_max=float(sp_obs.max()),
        inflight_share_graded=float(out["inflight"][g].mean()),
        gap_sigma_median=float(np.median(out["sigma"][gap])) if gap.any() else 0.0,
        gap_sigma_p90=float(np.percentile(out["sigma"][gap], 90)) if gap.any() else 0.0,
        dropout={str(k): v for k, v in drop.items()},
    )
    one = drop.get(25)
    checks = [
        ("rejected share <= 5%", d["rejected_share"] <= 0.05),
        ("p99 filtered speed on observed frames <= 40 m/s", d["speed_p99"] <= 40.0),
        ("1-s dropout median error <= 3 m", one is not None and one["median"] <= 3.0),
        ("1-s p90 <= linear p90", one is not None and one["p90"] <= one["linear_p90"]),
        (
            "1-s median <= linear median + 0.15 m",
            one is not None and one["median"] <= one["linear_median"] + 0.15,
        ),
        (
            "1-s within 2 sigma in [0.75, 0.97]",
            one is not None and 0.75 <= one["within_2sigma"] <= 0.97,
        ),
    ]
    d["checks"] = {k: bool(v) for k, v in checks}
    d["pass"] = all(d["checks"].values())
    if verbose:
        print("[ballfilter] diagnostics")
        for k, v in d.items():
            if k in ("dropout", "checks", "pass"):
                continue
            print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
        print("  dropout (hidden real observations; error vs observed position, m):")
        for L, r in drop.items():
            if r is None:
                print(f"    {L:3d} frames: no windows")
                continue
            print(
                f"    {L:3d} frames ({L / FPS:.2f}s): n={r['windows']} coverage {r['coverage']:.3f} "
                f"median {r['median']:.3f} p90 {r['p90']:.3f} | linear median {r['linear_median']:.3f} "
                f"p90 {r['linear_p90']:.3f} | within 2 sigma {r['within_2sigma']:.3f} | NLL {r['nll']:.3f}"
            )
        for k, v in checks:
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print(f"  -> {'KEEP Kalman' if d['pass'] else 'KILL: use linear_fallback'}")
    return d


def save(out, path=OUT_PATH, meta=None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = dict(
        pos=out["pos"].astype(np.float32),
        vel=out["vel"].astype(np.float32),
        speed=out["speed"].astype(np.float32),
        sigma=out["sigma"].astype(np.float32),
        obs=out["obs"],
        rejected=out.get("rejected", np.zeros(len(out["obs"]), bool)),
        inflight=out["inflight"],
        segment=out["segment"].astype(np.int32),
    )
    np.savez_compressed(path, **arrs)
    h = hashlib.sha256()
    for k in sorted(arrs):
        h.update(np.ascontiguousarray(arrs[k]).tobytes())
    digest = h.hexdigest()
    if meta is not None:
        path.with_suffix(".json").write_text(
            json.dumps({**meta, "sha256_arrays": digest}, indent=2, default=float)
        )
    return digest


def load(path=OUT_PATH) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- synthetic
def synthetic_ball(n: int = 60_000, seed: int = 0, r: float = 0.15):
    """Truth path (still / dribble / pass segments, reflected at the lines),
    noisy observations, Markov gaps, isolated spikes and 3-frame spike runs."""
    rng = np.random.default_rng(seed)
    truth = np.empty((n, 2))
    p = np.array([52.0, 34.0])
    t = 0
    while t < n:
        mode = rng.choice(3, p=[0.2, 0.5, 0.3])
        dur = int(rng.integers(10, 150))
        spd = (0.0, rng.uniform(1, 5), rng.uniform(10, 25))[mode]
        ang = rng.uniform(0, 2 * np.pi)
        v = spd * np.array([np.cos(ang), np.sin(ang)]) / FPS
        for _ in range(min(dur, n - t)):
            p = p + v
            for ax, hi in ((0, PITCH_X), (1, PITCH_Y)):
                if p[ax] < 0 or p[ax] > hi:
                    v[ax] = -v[ax]
                    p[ax] = np.clip(p[ax], 0, hi)
            truth[t] = p
            t += 1
    obs = truth + rng.normal(0, r, truth.shape)
    vis = np.ones(n, bool)
    i = 0
    while i < n:
        i += int(rng.geometric(1 / 75))
        m = int(rng.geometric(1 / 50))
        vis[i : i + m] = False
        i += m
    obs[~vis] = np.nan
    vi = np.flatnonzero(vis)
    sp = rng.choice(vi, int(0.005 * len(vi)), replace=False)
    obs[sp] = np.c_[rng.uniform(0, PITCH_X, len(sp)), rng.uniform(0, PITCH_Y, len(sp))]
    spike_mask = np.zeros(n, bool)
    spike_mask[sp] = True
    for s in rng.choice(vi[(vi > 10) & (vi < n - 10)], 30, replace=False):
        off = rng.normal(0, 15, 2)
        for j in range(s, s + 3):
            if vis[j]:
                obs[j] = truth[j] + off
                spike_mask[j] = True
    harmful = spike_mask & (np.linalg.norm(obs - truth, axis=1) > 3.0)
    return obs.astype(np.float32), truth, harmful, r


def smoke() -> bool:
    t0 = time.time()
    obs, truth, harmful, r_true = synthetic_ball()
    half = np.zeros(len(obs), np.int8)
    vis = np.isfinite(obs).all(1)
    r_hat = estimate_r(obs, visible_onpitch(obs))
    kept, off, spike, _ = gate(obs, half, d0_for(r_hat))
    rej = ~kept & vis
    recall = float(rej[harmful].mean())
    false_rej = float(rej[vis & ~harmful].mean())
    print(
        f"[smoke] numba {HAVE_NUMBA}; harmful spikes {int(harmful.sum())}, recall {recall:.3f}; "
        f"false rejection {false_rej:.4f}; r_hat {r_hat:.3f} (true {r_true})"
    )
    q, table = select_q(obs, half, r_hat)
    out = pipeline(obs, half, q, r_hat)
    drop = dropout_test(obs, half, q, r_hat)
    d = diagnostics(obs, half, out, drop)
    hid_err = np.linalg.norm(out["pos"] - truth, axis=1)
    gap = np.isfinite(hid_err) & ~vis
    print(
        f"[smoke] error vs TRUE path: observed frames median {np.nanmedian(hid_err[out['obs']]):.3f} m "
        f"(raw noise median {np.median(np.linalg.norm(obs[out['obs']] - truth[out['obs']], axis=1)):.3f}); "
        f"gap frames median {np.median(hid_err[gap]):.3f} m"
    )
    out2 = pipeline(obs, half, q, r_hat)
    det = bool(np.array_equal(np.nan_to_num(out["pos"]), np.nan_to_num(out2["pos"])))
    checks = [
        ("harmful spike recall >= 0.9", recall >= 0.9),
        ("false rejection <= 1%", false_rej <= 0.01),
        ("r_hat within 30% of truth", abs(r_hat - r_true) <= 0.3 * r_true),
        (
            "observed-frame error below raw noise",
            np.nanmedian(hid_err[out["obs"]])
            < np.median(np.linalg.norm(obs[out["obs"]] - truth[out["obs"]], axis=1)),
        ),
        ("diagnostics kill criteria pass", d["pass"]),
        ("deterministic", det),
    ]
    ok = True
    for k, v in checks:
        ok &= bool(v)
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"[smoke] {time.time() - t0:.1f}s -> {'PASS' if ok else 'FAIL'}")
    return ok


def run(q=None, out_path=OUT_PATH) -> int:
    from mtah.frames import check_games, load_tracking

    t0 = time.time()
    check_games()
    cache = load_tracking()
    ball = cache["ball"]
    half = half_index(len(ball))
    r = estimate_r(ball, visible_onpitch(ball))
    print(
        f"[ballfilter] r estimated {r:.3f} m; jump allowance d0 {d0_for(r):.3f} m; numba {HAVE_NUMBA}"
    )
    table = {}
    if q is None:
        print(
            "[ballfilter] q selection (pre-registered: min mean NLL of hidden 1-s observations)"
        )
        q, table = select_q(ball, half, r)
    print(f"[ballfilter] q = {q}")
    out = pipeline(ball, half, q, r)
    drop = dropout_test(ball, half, q, r)
    d = diagnostics(ball, half, out, drop)
    if not d["pass"]:
        print("[ballfilter] kill criterion failed: saving linear fallback instead")
        out = linear_fallback(ball, out["obs"])
        out["rejected"] = np.isfinite(ball).all(1) & ~out["obs"]
        out["rejected"] = np.isfinite(ball).all(1) & ~out["obs"]
    digest = save(
        out,
        out_path,
        meta=dict(
            module="ballfilter",
            mode="kalman" if d["pass"] else "linear_fallback",
            q_table={str(k): v for k, v in table.items()},
            diag=d,
            seconds=time.time() - t0,
        ),
    )
    print(
        f"[ballfilter] wrote {out_path} (+ .json)  arrays sha256 {digest[:16]}  ({time.time() - t0:.1f}s)"
    )
    return 0 if d["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("run")
    p.add_argument("--q", type=float, default=None, help="skip the grid search")
    p.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        return 0 if smoke() else 1
    return run(a.q, a.out)


if __name__ == "__main__":
    sys.exit(main())
