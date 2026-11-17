#!/usr/bin/env python3
"""mtah/evidence.py -- step 2: ball-proximity possession evidence.

Per frame, from the step-1 ball track and on-pitch players:
  llr  = log[(sum_home K(d) + eps) / (sum_away K(d) + eps)],  K(d) = exp(-d^2 / 2s^2),
         s^2 = S0^2 + sigma^2 (ball uncertainty widens the kernel), eps = K(DC):
         a "loose ball" term, so when nobody is within ~DC metres the evidence
         goes to 0 instead of exploding. Clipped to +-CAP nats.
         Multiplied by W_FLIGHT while the ball is fast (in flight / passed),
         so a pass travelling past opponents does not flip possession.
         Zero where there is no ball estimate and before each half's kick-off.
  llr_scaled = llr / tau, tau = integrated autocorrelation time (frames) of the
         within-possession residual of llr (llr minus its 5-s local mean), with
         Sokal's automatic window. Consecutive frames repeat the same
         information; summing raw llr overcounts it by about tau.
  pseudo = high-confidence labels from OBSERVED ball frames (not interpolated):
         nearest on-pitch player < 1.5 m, other team's nearest at least
         MARGIN farther, ball speed < SPEED, sigma < 0.5 m, after kick-off.
         Pre-registered yield rule: if pseudo frames < 5% of observed in-play
         graded frames, loosen margin 2.0 -> 1.5 m; if still < 5%, speed 6 -> 8 m/s;
         if still < 5%, T2 (geometry channel) is killed.
  play  = graded frames at or after each half's kick-off. Kick-off = the
         "Kick Off (Start)" tag minus the tag-vs-formation offset measured in
         frames.kickoff_alignment (label-free); block start if the alignment
         check fails or no start tag lies near the block start.

Assumptions:
  E1  Players more than 2 m outside the pitch (bench, warm-up) are excluded.
  E2  Possession is held by the team of the player nearest the ball; the
      tracklet identity is never used across frames.
  E3  The scorer's frames and ours coincide (verified in step 0).

Outputs cache/evidence.npz (n = 199,319): llr, llr_scaled, conf (f32);
d_home, d_away (f32 m, nearest on-pitch player, NaN if none); pseudo (i8 -1/0/1);
play (bool); plus cache/evidence.json (thresholds, yield, tau, diagnostics).

CLI (repo root):
  python -m mtah.evidence smoke     # synthetic match with known truth
  python -m mtah.evidence run       # needs cache/ballfilter.npz from step 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import logsumexp

from mtah.frames import BLOCKS, FPS, N_FRAMES, PITCH_X, PITCH_Y, ROOT, half_index, runs

S0 = 1.0  # m, kernel width
DC = 3.0  # m, loose-ball radius
CAP = 4.0  # nats
W_FLIGHT = 0.1  # evidence weight while the ball is in flight
OFFPITCH_M = 2.0  # m outside the lines before a player is excluded
PSEUDO0 = dict(dist=1.5, margin=2.0, speed=6.0, sigma=0.5)
LOOSEN = (dict(margin=1.5), dict(speed=8.0))
YIELD_MIN = 0.05
TAU_SMOOTH = 125  # frames, local-mean window for the residual
TAU_KMAX = 750
OUT_PATH = ROOT / "cache" / "evidence.npz"
GAP_BINS = ((0, 0.5), (0.5, 3), (3, 10), (10, 30), (30, np.inf))


# --------------------------------------------------------------------------- play mask
def play_starts(cache, tags_path=None, verbose=True) -> list:
    """First in-play frame per graded block (see module docstring)."""
    from mtah.frames import (
        attack_direction,
        default_tags_path,
        kickoff_alignment,
        kickoff_events,
        load_tag_restarts,
    )

    tags = load_tag_restarts(tags_path or default_tags_path(), verbose=False)
    ko = [(f, k) for f, k in tags if k.startswith("kick_off")]
    signs = attack_direction(
        cache["players"], cache["team"], half_index(), verbose=False
    )
    ev = kickoff_events(
        cache["players"], cache["team"], cache["ball"], signs, verbose=False
    )
    al = kickoff_alignment(ko, ev, verbose=False)
    starts = []
    for a, b in BLOCKS:
        st = [
            f for f, k in ko if k == "kick_off_start" and a - 2000 <= f <= a + 300 * FPS
        ]
        if al["pass"] and st:
            est = int(round(st[0] - al["semantic_offset_frames"]))
        else:
            est = a
        starts.append(int(min(max(est, a), b)))
    if verbose:
        print(
            f"[play] alignment pass {al['pass']}, tag-vs-formation offset {al['semantic_offset_frames']} frames"
        )
        for (a, _), s in zip(BLOCKS, starts):
            print(
                f"  block start {a}  ->  play starts {s}  ({(s - a) / FPS:.1f} s pre-kick-off)"
            )
    return starts


def play_mask(starts, n=N_FRAMES) -> np.ndarray:
    m = np.zeros(n, bool)
    for (a, b), s in zip(BLOCKS, starts):
        m[s : b + 1] = True
    return m


# --------------------------------------------------------------------------- evidence
def onpitch(players, margin=OFFPITCH_M) -> np.ndarray:
    x, y = players[..., 0], players[..., 1]
    return (
        np.isfinite(x)
        & (x >= -margin)
        & (x <= PITCH_X + margin)
        & (y >= -margin)
        & (y <= PITCH_Y + margin)
    )


def ball_evidence(
    players, team, pos, sigma, inflight, play, chunk: int = 40_000
) -> dict:
    n = len(pos)
    valid = onpitch(players)
    home, away = team == 0, team == 1
    llr = np.zeros(n, np.float32)
    dh = np.full(n, np.nan, np.float32)
    da = np.full(n, np.nan, np.float32)
    have = np.isfinite(pos).all(1) & np.isfinite(sigma)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d2 = ((players[s:e] - pos[s:e, None, :]) ** 2).sum(-1).astype(np.float64)
        d2 = np.where(valid[s:e], d2, np.inf)
        s2 = (S0**2 + np.where(have[s:e], sigma[s:e], np.inf) ** 2)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            lk = -d2 / (2 * s2)
            lh = logsumexp(np.where(home, lk, -np.inf), axis=1)
            la = logsumexp(np.where(away, lk, -np.inf), axis=1)
            eps = -(DC**2) / (2 * s2[:, 0])
            l = np.logaddexp(lh, eps) - np.logaddexp(la, eps)
            mh = np.sqrt(np.min(np.where(home, d2, np.inf), axis=1))
            ma = np.sqrt(np.min(np.where(away, d2, np.inf), axis=1))
        ok = have[s:e] & np.isfinite(l)
        llr[s:e] = np.where(ok, np.clip(l, -CAP, CAP), 0.0)
        dh[s:e] = np.where(have[s:e] & np.isfinite(mh), mh, np.nan)
        da[s:e] = np.where(have[s:e] & np.isfinite(ma), ma, np.nan)
    llr = np.where(inflight, W_FLIGHT * llr, llr)
    llr = np.where(play, llr, 0.0).astype(np.float32)
    return dict(
        llr=llr,
        d_home=dh,
        d_away=da,
        excluded_share=float(1 - valid[np.isfinite(players[..., 0])].mean()),
    )


def pseudo_labels(d_home, d_away, speed, sigma, obs, play, th) -> np.ndarray:
    near = np.fmin(d_home, d_away)
    other = np.fmax(d_home, d_away)
    with np.errstate(invalid="ignore"):
        ok = (
            obs
            & play
            & (near < th["dist"])
            & (other - near >= th["margin"])
            & (speed < th["speed"])
            & (sigma < th["sigma"])
        )
    return np.where(ok, np.where(d_home < d_away, 0, 1), -1).astype(np.int8)


def pseudo_with_yield_rule(d_home, d_away, speed, sigma, obs, play, verbose=True):
    """Pre-registered: start at PSEUDO0, apply LOOSEN steps in order while yield < 5%."""
    denom = max(int((obs & play).sum()), 1)
    th = dict(PSEUDO0)
    steps = [dict(th)]
    y = pseudo_labels(d_home, d_away, speed, sigma, obs, play, th)
    yld = float((y >= 0).sum() / denom)
    log = [(dict(th), yld)]
    for change in LOOSEN:
        if yld >= YIELD_MIN:
            break
        th.update(change)
        y = pseudo_labels(d_home, d_away, speed, sigma, obs, play, th)
        yld = float((y >= 0).sum() / denom)
        log.append((dict(th), yld))
    alive = yld >= YIELD_MIN
    if verbose:
        print(
            "[pseudo] yield rule (denominator: observed in-play graded frames "
            f"= {denom})"
        )
        for t, v in log:
            print(
                f"  dist<{t['dist']} margin>={t['margin']} speed<{t['speed']} sigma<{t['sigma']}: yield {v:.4f}"
            )
        print(
            f"  {'PASS' if alive else 'KILL'}  final yield {yld:.4f} "
            f"{'>=' if alive else '<'} {YIELD_MIN} -> T2 geometry channel {'alive' if alive else 'KILLED'}"
        )
    return y, th, yld, alive, log


# --------------------------------------------------------------------------- tau
def act_time(x, mask, smooth=TAU_SMOOTH, kmax=TAU_KMAX, c=5.0) -> dict:
    """Integrated autocorrelation time (frames) of the residual x - local mean,
    over masked frames, Sokal automatic window M >= c * tau(M)."""
    w = mask.astype(np.float64)
    xm = np.where(mask, x, 0.0).astype(np.float64)
    num = uniform_filter1d(xm, smooth, mode="constant")
    den = uniform_filter1d(w, smooth, mode="constant")
    res = np.where(mask, xm - num / np.maximum(den, 1e-9), 0.0)
    n = len(res)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    fr = np.fft.rfft(res, nfft)
    fw = np.fft.rfft(w, nfft)
    ac = np.fft.irfft(fr * np.conj(fr), nfft)[: kmax + 1]
    cnt = np.fft.irfft(fw * np.conj(fw), nfft)[: kmax + 1]
    cov = ac / np.maximum(cnt, 1.0)
    rho = cov / cov[0]
    tau, M = 1.0, kmax
    for m in range(1, kmax + 1):
        tau = 1.0 + 2.0 * float(rho[1 : m + 1].sum())
        if m >= c * tau:
            M = m
            break
    return dict(
        tau=float(np.clip(tau, 1.0, 250.0)),
        window=int(M),
        rho_1=float(rho[1]),
        rho_25=float(rho[25]),
        rho_125=float(rho[min(125, kmax)]),
    )


# --------------------------------------------------------------------------- diagnostics
def gap_histogram(have, play) -> list:
    rows = []
    lens = []
    for a, b in BLOCKS:
        s, e, v = runs((have[a : b + 1] | ~play[a : b + 1]).astype(np.uint8))
        lens += list((e - s)[v == 0])
    lens = np.array(lens) / FPS
    tot = max(lens.sum(), 1e-9)
    for lo, hi in GAP_BINS:
        m = (lens > lo) & (lens <= hi)
        rows.append(
            dict(
                bin=f"({lo},{hi}] s",
                gaps=int(m.sum()),
                missing_share=float(lens[m].sum() / tot),
            )
        )
    return rows


def compute(cache, bf, starts, verbose=True) -> dict:
    t0 = time.time()
    n = len(cache["ball"])
    g = half_index(n) >= 0
    play = play_mask(starts, n)
    ev = ball_evidence(
        cache["players"], cache["team"], bf["pos"], bf["sigma"], bf["inflight"], play
    )
    y, th, yld, alive, ylog = pseudo_with_yield_rule(
        ev["d_home"], ev["d_away"], bf["speed"], bf["sigma"], bf["obs"], play, verbose
    )
    evid = play & (ev["llr"] != 0)
    ta = act_time(ev["llr"], evid)
    llr = ev["llr"]
    out = dict(
        llr=llr,
        llr_scaled=(llr / ta["tau"]).astype(np.float32),
        conf=(np.abs(llr) / CAP).astype(np.float32),
        d_home=ev["d_home"],
        d_away=ev["d_away"],
        pseudo=y,
        play=play,
    )
    have = np.isfinite(bf["pos"]).all(1)
    gp = g & play
    diag = dict(
        play_starts=starts,
        play_graded_share=float(play[g].mean()),
        offpitch_player_frames_excluded=ev["excluded_share"],
        evidence_share_play=float(evid[gp].mean()),
        strong_share_play=float((np.abs(llr) >= 1.0)[gp].mean()),
        inflight_share_play=float(bf["inflight"][gp].mean()),
        llr_home_share=float((llr[evid] > 0).mean()) if evid.any() else 0.0,
        pseudo_thresholds=th,
        pseudo_yield=yld,
        pseudo_log=ylog,
        t2_alive=bool(alive),
        pseudo_n=int((y >= 0).sum()),
        pseudo_home_share=float((y[y >= 0] == 0).mean()) if (y >= 0).any() else 0.0,
        pseudo_per_half=[int(((y >= 0) & (half_index(n) == k)).sum()) for k in (0, 1)],
        tau=ta,
        gaps=gap_histogram(have, play),
        seconds=time.time() - t0,
    )
    if verbose:
        print("[evidence] diagnostics")
        for k, v in diag.items():
            if k in ("pseudo_log", "gaps", "tau"):
                continue
            print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")
        print(
            f"  tau {ta['tau']:.2f} frames (window {ta['window']}; rho1 {ta['rho_1']:.3f} "
            f"rho25 {ta['rho_25']:.3f} rho125 {ta['rho_125']:.3f})"
        )
        print("  ball-missing gaps in play (count, share of missing frames):")
        for r in diag["gaps"]:
            print(f"    {r['bin']:12s} {r['gaps']:6d}  {r['missing_share']:.3f}")
    return out, diag


def save(out, diag, path=OUT_PATH) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    h = hashlib.sha256()
    for k in sorted(out):
        h.update(np.ascontiguousarray(out[k]).tobytes())
    digest = h.hexdigest()
    path.with_suffix(".json").write_text(
        json.dumps({**diag, "sha256_arrays": digest}, indent=2, default=float)
    )
    return digest


def load(path=OUT_PATH) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- smoke / run
def smoke() -> bool:
    from mtah.ballfilter import pipeline
    from mtah.frames import synthetic_match

    t0 = time.time()
    cache, truth, _ = synthetic_match(seed=0)
    # a bench player beside the touchline must be ignored
    cache["players"][:, 45] = np.c_[
        np.full(N_FRAMES, 50.0), np.full(N_FRAMES, PITCH_Y + 4.0)
    ]
    bf = pipeline(cache["ball"], half_index(), q=100.0)
    starts = [a for a, _ in BLOCKS]
    out, diag = compute(cache, bf, starts)
    g = (half_index() >= 0) & (truth < 2)
    y = out["pseudo"]
    sel = (y >= 0) & g
    prec = float((y[sel] == truth[sel]).mean()) if sel.any() else 0.0
    strong = g & (np.abs(out["llr"]) >= 1.0)
    sign_acc = float(((out["llr"][strong] > 0) == (truth[strong] == 0)).mean())
    out2, _ = compute(cache, bf, starts, verbose=False)
    det = all(np.array_equal(out[k], out2[k], equal_nan=True) for k in out)
    checks = [
        ("pseudo-label precision vs truth >= 0.95", prec >= 0.95),
        ("pseudo yield >= 5% (T2 alive)", diag["t2_alive"]),
        ("strong-evidence sign accuracy >= 0.90", sign_acc >= 0.90),
        (
            "off-pitch bench player excluded",
            diag["offpitch_player_frames_excluded"] > 0.0,
        ),
        ("tau in [1, 250]", 1.0 <= diag["tau"]["tau"] <= 250.0),
        ("deterministic", det),
    ]
    print(f"[smoke] pseudo precision {prec:.4f}; strong sign accuracy {sign_acc:.4f}")
    ok = True
    for k, v in checks:
        ok &= bool(v)
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"[smoke] {time.time() - t0:.1f}s -> {'PASS' if ok else 'FAIL'}")
    return ok


def run(out_path=OUT_PATH) -> int:
    from mtah import ballfilter
    from mtah.frames import check_games, load_tracking, soft_evidence

    t0 = time.time()
    check_games()
    cache = load_tracking()
    if not ballfilter.OUT_PATH.is_file():
        raise FileNotFoundError(
            f"{ballfilter.OUT_PATH} missing: run python -m mtah.ballfilter run first"
        )
    bf = ballfilter.load()
    meta = json.loads(ballfilter.OUT_PATH.with_suffix(".json").read_text())
    print(
        f"[evidence] ball track: {meta.get('mode')} (sha {meta.get('sha256_arrays', '')[:16]})"
    )
    starts = play_starts(cache)
    out, diag = compute(cache, bf, starts)
    # sanity only (not validation): sign agreement with level0's raw evidence
    z0 = soft_evidence(cache["players"], cache["ball"], cache["team"])
    both = (out["llr"] != 0) & (z0 != 0) & out["play"]
    agree = (
        float(((out["llr"][both] > 0) == (z0[both] > 0)).mean())
        if both.any()
        else float("nan")
    )
    diag["sign_agreement_with_level0"] = agree
    print(
        f"  sign agreement with level0 raw evidence (sanity) {agree:.4f} on {int(both.sum())} frames"
    )
    diag["ballfilter_mode"] = meta.get("mode")
    digest = save(out, diag, out_path)
    print(
        f"[evidence] wrote {out_path} (+ .json)  arrays sha256 {digest[:16]}  ({time.time() - t0:.1f}s)"
    )
    return 0 if diag["t2_alive"] else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("run")
    p.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        return 0 if smoke() else 1
    return run(a.out)


if __name__ == "__main__":
    sys.exit(main())
