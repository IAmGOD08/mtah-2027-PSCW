#!/usr/bin/env python3
"""mtah/deadball2.py -- step 4: learned dead-ball evidence.

Anchors (algorithmic, graded frames only; no human labels):
  dead  1 ball out of bounds: observed ball > 0.5 m beyond a line
        2 pre-kick-off: graded frames before each half's kick-off (step 2)
        3 kick-off formation: the 2 s before each detected kick-off event
        4 placed ball: observed ball static (< 0.5 m/s) for >= 1 s within
          +-10 s of a non-kick-off restart tag
  live  observed ball inside the pitch (>= 1 m from the lines), after kick-off,
        and either fast (> 8 m/s, kicked) or at a player's feet and moving
        (step-2 pseudo-label with speed > 1.5 m/s); never within 2 s of a dead
        anchor.
Model: HistGradientBoosting on PLAYER-ONLY features (speed distribution,
  running shares, dispersion, team-centroid distance, own-half share, visible
  count; each also as centred 1 s / 3 s / 9 s rolling means). Ball features
  are excluded on purpose: every anchor is defined by the ball, so a ball-aware
  model would relearn the anchor rules and say nothing on the ~41% of frames
  without a ball. Classes balanced, cross-fit by half (train on one half,
  predict the other), isotonic calibration on the pooled out-of-fold scores:
  logit(p) is then a likelihood ratio at prior odds 1.
Ball enters as bounded rules added to the model log-odds:
  +3 observed out of bounds; +1 placed ball (static >= 1 s); -3 observed in
  the pitch and faster than 8 m/s; -3 observed in the pitch, within 1.5 m of
  an on-pitch player and moving faster than 2 m/s (ball under control, live);
  pre-kick-off forced to +6. Clipped to +-6.
Revision 1 (declared after the first real run, before re-running, no
  further revisions): added the ball-control rule. The first run showed that
  player-only features cannot tell slow build-up from a stoppage (12% of
  moving-ball pseudo frames called dead); only the ball can. Thresholds,
  anchors and model unchanged. The control rule overlaps the moving-ball
  check above 2 m/s, so the dead-ball bands and step 5's D-on-pseudo-frames
  share are the independent checks.
  dead_llr_scaled = dead_llr / tau_d (integrated autocorrelation time).

Kill criteria (pre-registered; failure -> exit 1 and rules-only evidence):
  out-of-fold AUC, out-of-bounds anchors vs live anchors >= 0.85;
  quick decode (2-s smoothed llr + prior 0.25, runs >= 2 s) dead fraction in
  [0.15, 0.35] and median dead stretch in [8, 40] s;
  <= 3% of moving-ball pseudo-label frames called dead by the quick decode.
The quick decode is a diagnostic only; final D comes from the HSMM (step 5).

Outputs cache/deadball2.npz (n = 199,319): dead_llr, dead_llr_scaled,
p_model (calibrated, player-only) f32; anchors i8 (-1 none, 0 live, 1 dead);
anchor_kind i8 (0 live, 1-4 as above, -1 none); quick_dead bool;
plus cache/deadball2.json.

CLI (repo root):
  python -m mtah.deadball2 smoke
  python -m mtah.deadball2 run      # needs cache/ballfilter.npz and cache/evidence.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from mtah.frames import (
    BANDS,
    BLOCKS,
    FPS,
    N_FRAMES,
    PITCH_X,
    PITCH_Y,
    ROOT,
    half_index,
    runs,
)

OOB_M = 0.5
INSIDE_M = 1.0
FAST = 8.0
STATIC = 0.5
STATIC_RUN = 25
TAG_WIN = 250
FORMATION = 50
NEG_GUARD = 50
RULE_OOB, RULE_PLACED, RULE_FAST, RULE_PREKICK, CLIP = 3.0, 1.0, -3.0, 6.0, 6.0
RULE_CONTROL, CONTROL_DIST, CONTROL_SPEED = -3.0, 1.5, 2.0
REVISION = 1
P_CLIP = (0.01, 0.99)
PRIOR = 0.25
MIN_RUN = 50
AUC_MIN = 0.85
MOVING_DEAD_MAX = 0.03
SUBSAMPLE = 5
HGB = dict(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    min_samples_leaf=40,
    early_stopping=False,
    random_state=0,
)
OUT_PATH = ROOT / "cache" / "deadball2.npz"
KINDS = {1: "out_of_bounds", 2: "pre_kickoff", 3: "kickoff_formation", 4: "placed_ball"}


# --------------------------------------------------------------------------- features
def _nanroll(x, w):
    f = np.isfinite(x)
    num = uniform_filter1d(np.where(f, x, 0.0), w, mode="nearest")
    den = uniform_filter1d(f.astype(np.float64), w, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-9, num / den, np.nan)


def player_features(players, team, signs) -> tuple[np.ndarray, list]:
    """(n, k) float32 features from players only."""
    from mtah.evidence import onpitch

    n = len(players)
    valid = onpitch(players)
    P = np.where(valid[..., None], players, np.nan)
    sp = np.full((n, P.shape[1]), np.nan)
    sp[1:] = np.linalg.norm(np.diff(P, axis=0), axis=2) * FPS
    sp[sp > 12] = np.nan
    fin = np.isfinite(sp)
    cnt = np.maximum(fin.sum(1), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(sp, axis=1)
        p90 = np.nanpercentile(sp, 90, axis=1)
        sx = np.nanstd(P[..., 0], axis=1)
        sy = np.nanstd(P[..., 1], axis=1)
        ch = np.nanmean(np.where((team == 0)[None, :, None], P, np.nan), axis=1)
        ca = np.nanmean(np.where((team == 1)[None, :, None], P, np.nan), axis=1)
    run2 = (np.nan_to_num(sp) > 2).sum(1) / cnt
    run4 = (np.nan_to_num(sp) > 4).sum(1) / cnt
    run2[fin.sum(1) == 0] = np.nan
    run4[fin.sum(1) == 0] = np.nan
    cdist = np.linalg.norm(ch - ca, axis=1)
    mid = (BLOCKS[0][1] + BLOCKS[1][0]) // 2
    sign = np.where(np.arange(n) < mid, signs[0], signs[1])[:, None]
    x = P[..., 0]
    cx = PITCH_X / 2
    own = np.where(
        team == 0,
        np.where(sign > 0, x < cx, x > cx),
        np.where(sign > 0, x > cx, x < cx),
    )
    vis = np.isfinite(x)
    own_share = np.where(
        vis.sum(1) > 0, (own & vis).sum(1) / np.maximum(vis.sum(1), 1), np.nan
    )
    nvis = vis.sum(1).astype(np.float64)
    base = dict(
        speed_med=med,
        speed_p90=p90,
        run2=run2,
        run4=run4,
        spread_x=sx,
        spread_y=sy,
        centroid_dist=cdist,
        own_half=own_share,
        n_visible=nvis,
    )
    cols, names = [], []
    for k, v in base.items():
        cols.append(v)
        names.append(k)
        if k in (
            "speed_med",
            "speed_p90",
            "run2",
            "run4",
            "spread_x",
            "spread_y",
            "own_half",
        ):
            for w in (25, 75, 225):
                cols.append(_nanroll(v, w))
                names.append(f"{k}_r{w}")
    return np.column_stack(cols).astype(np.float32), names


# --------------------------------------------------------------------------- anchors
def placed_ball(obs, speed, tags) -> np.ndarray:
    n = len(obs)
    st = obs & (np.nan_to_num(speed, nan=99.0) < STATIC)
    out = np.zeros(n, bool)
    s, e, v = runs(st.astype(np.uint8))
    for a, b in zip(s[v == 1], e[v == 1]):
        if b - a >= STATIC_RUN:
            out[a:b] = True
    near = np.zeros(n, bool)
    for f, k in tags:
        if not k.startswith("kick_off"):
            near[max(0, f - TAG_WIN) : min(n, f + TAG_WIN + 1)] = True
    return out & near, out


def make_anchors(
    ball_pos, obs, speed, pseudo, play, kick_events, tags
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(obs)
    g = half_index(n) >= 0
    x, y = ball_pos[:, 0], ball_pos[:, 1]
    with np.errstate(invalid="ignore"):
        oob = obs & (
            (x < -OOB_M) | (x > PITCH_X + OOB_M) | (y < -OOB_M) | (y > PITCH_Y + OOB_M)
        )
        inside = (
            obs
            & (x > INSIDE_M)
            & (x < PITCH_X - INSIDE_M)
            & (y > INSIDE_M)
            & (y < PITCH_Y - INSIDE_M)
        )
    prek = g & ~play
    form = np.zeros(n, bool)
    for e in kick_events:
        form[max(0, e - FORMATION) : e] = True
    placed, static_any = placed_ball(obs, speed, tags)
    kind = np.full(n, -1, np.int8)
    for k, m in ((4, placed), (3, form), (1, oob), (2, prek)):
        kind[m & g] = k
    dead = kind > 0
    guard = (
        uniform_filter1d(dead.astype(np.float64), 2 * NEG_GUARD + 1, mode="constant")
        > 0
    )
    sp = np.nan_to_num(speed, nan=0.0)
    live = (
        g
        & play
        & inside
        & ((sp > FAST) | ((pseudo >= 0) & (sp > 1.5)))
        & ~guard
        & ~dead
    )
    kind[live] = 0
    lab = np.where(kind > 0, 1, np.where(kind == 0, 0, -1)).astype(np.int8)
    counts = {KINDS[k]: int((kind == k).sum()) for k in KINDS}
    counts["live"] = int(live.sum())
    return lab, kind, dict(counts=counts, oob=oob, inside=inside, static_any=static_any)


# --------------------------------------------------------------------------- model
def cross_fit(X, lab, n) -> tuple[np.ndarray, dict]:
    h = half_index(n)
    raw = np.full(n, np.nan)
    info = {}
    for k in (0, 1):
        tr = np.flatnonzero((lab >= 0) & (h == 1 - k))[::SUBSAMPLE]
        yk = lab[tr]
        w = np.where(
            yk == 1, 0.5 / max((yk == 1).sum(), 1), 0.5 / max((yk == 0).sum(), 1)
        ) * len(yk)
        clf = HistGradientBoostingClassifier(**HGB)
        clf.fit(X[tr], yk, sample_weight=w)
        te = np.flatnonzero(h == k)
        raw[te] = clf.predict_proba(X[te])[:, 1]
        info[f"train_half_{1 - k}"] = dict(n=int(len(tr)), pos=int((yk == 1).sum()))
    return raw, info


def calibrate(raw, lab) -> tuple[np.ndarray, IsotonicRegression]:
    sel = np.flatnonzero(lab >= 0)
    y = lab[sel]
    w = np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1))
    iso = IsotonicRegression(y_min=P_CLIP[0], y_max=P_CLIP[1], out_of_bounds="clip")
    iso.fit(raw[sel], y, sample_weight=w)
    p = np.full(len(raw), np.nan)
    ok = np.isfinite(raw)
    p[ok] = iso.predict(raw[ok])
    return p, iso


def logit(p):
    p = np.clip(p, *P_CLIP)
    return np.log(p / (1 - p))


def auc_by_kind(raw, lab, kind) -> dict:
    out = {}
    neg = kind == 0
    for k, name in KINDS.items():
        pos = kind == k
        if pos.sum() >= 50 and neg.sum() >= 50:
            sel = pos | neg
            out[name] = float(roc_auc_score(lab[sel], raw[sel]))
        else:
            out[name] = None
    sel = lab >= 0
    out["all"] = (
        float(roc_auc_score(lab[sel], raw[sel]))
        if (lab[sel] == 1).any() and (lab[sel] == 0).any()
        else None
    )
    return out


def quick_decode(dead_llr, n=N_FRAMES) -> np.ndarray:
    g = half_index(n) >= 0
    sm = uniform_filter1d(dead_llr.astype(np.float64), MIN_RUN, mode="nearest") + logit(
        np.array(PRIOR)
    )
    dead = (sm > 0) & g
    for val in (0, 1):  # remove short runs of either value, live first
        for a, b in BLOCKS:
            s, e, v = runs(dead[a : b + 1].astype(np.uint8))
            for ss, ee, vv in zip(s, e, v):
                if vv == val and ee - ss < MIN_RUN:
                    dead[a + ss : a + ee] = not val
    return dead & g


def dead_stats(dead, play) -> dict:
    g = (half_index(len(dead)) >= 0) & play
    lens = []
    for a, b in BLOCKS:
        s, e, v = runs(dead[a : b + 1].astype(np.uint8))
        lens += list((e - s)[v == 1] / FPS)
    return dict(
        dead_fraction=float(dead[g].mean()),
        stretches=len(lens),
        median_stretch_s=float(np.median(lens)) if lens else 0.0,
    )


# --------------------------------------------------------------------------- pipeline
def compute(cache, bf, ev, tags, kick_events, verbose=True) -> tuple[dict, dict]:
    from mtah.evidence import act_time
    from mtah.frames import attack_direction

    t0 = time.time()
    n = len(cache["ball"])
    g = half_index(n) >= 0
    play = ev["play"].astype(bool)
    signs = attack_direction(
        cache["players"], cache["team"], half_index(n), verbose=False
    )
    X, names = player_features(cache["players"], cache["team"], signs)
    lab, kind, an = make_anchors(
        bf["pos"],
        bf["obs"].astype(bool),
        bf["speed"],
        ev["pseudo"],
        play,
        kick_events,
        tags,
    )
    raw, fitinfo = cross_fit(X, lab, n)
    p, _ = calibrate(raw, lab)
    aucs = auc_by_kind(raw, lab, kind)
    sp = np.nan_to_num(bf["speed"], nan=0.0)
    rules = np.zeros(n)
    rules += np.where(an["oob"], RULE_OOB, 0.0)
    rules += np.where(an["static_any"], RULE_PLACED, 0.0)
    rules += np.where(an["inside"] & (sp > FAST), RULE_FAST, 0.0)
    near = np.fmin(ev["d_home"], ev["d_away"])
    with np.errstate(invalid="ignore"):
        control = (
            an["inside"] & (near < CONTROL_DIST) & (sp > CONTROL_SPEED) & (sp <= FAST)
        )
    rules += np.where(control, RULE_CONTROL, 0.0)
    model_llr = np.where(np.isfinite(p), logit(np.nan_to_num(p, nan=0.5)), 0.0)
    killed_model = False
    dead_llr = np.clip(model_llr + rules, -CLIP, CLIP)
    dead_llr = np.where(g & ~play, RULE_PREKICK, dead_llr)
    dead_llr = np.where(g, dead_llr, 0.0)
    qd = quick_decode(dead_llr, n)
    ds = dead_stats(qd, play)
    moving = g & play & (ev["pseudo"] >= 0) & (sp > 1.5)
    moving_dead = float(qd[moving].mean()) if moving.any() else 0.0
    auc_oob = aucs.get("out_of_bounds")
    checks = [
        (
            f"OOF AUC out-of-bounds vs live >= {AUC_MIN}",
            auc_oob is not None and auc_oob >= AUC_MIN,
        ),
        (
            f"quick dead fraction in {BANDS['dead_fraction']}",
            BANDS["dead_fraction"][0]
            <= ds["dead_fraction"]
            <= BANDS["dead_fraction"][1],
        ),
        (
            f"quick median dead stretch in {BANDS['median_dead_s']} s",
            BANDS["median_dead_s"][0]
            <= ds["median_stretch_s"]
            <= BANDS["median_dead_s"][1],
        ),
        (
            f"moving-ball pseudo frames called dead <= {MOVING_DEAD_MAX}",
            moving_dead <= MOVING_DEAD_MAX,
        ),
    ]
    passed = all(v for _, v in checks)
    if not passed:
        killed_model = True
        dead_llr = np.clip(rules, -CLIP, CLIP)
        dead_llr = np.where(g & ~play, RULE_PREKICK, dead_llr)
        dead_llr = np.where(g, dead_llr, 0.0)
        qd = quick_decode(dead_llr, n)
    ta = act_time(dead_llr, g & play, smooth=375)
    have = np.isfinite(bf["pos"]).all(1)
    gap_p = {}
    for lo, hi in ((0, 3), (3, 10), (10, 30), (30, 1e9)):
        m = np.zeros(n, bool)
        for a, b in BLOCKS:
            s, e, v = runs((have[a : b + 1] | ~play[a : b + 1]).astype(np.uint8))
            for ss, ee, vv in zip(s, e, v):
                if vv == 0 and lo * FPS < ee - ss <= hi * FPS:
                    m[a + ss : a + ee] = True
        gap_p[f"({lo},{hi if hi < 1e9 else 'inf'}] s"] = (
            float(np.nanmean(p[m])) if m.any() else None,
            int(m.sum()),
        )
    vis_play = g & play & have
    out = dict(
        dead_llr=dead_llr.astype(np.float32),
        dead_llr_scaled=(dead_llr / ta["tau"]).astype(np.float32),
        p_model=np.nan_to_num(p, nan=0.5).astype(np.float32),
        anchors=lab,
        anchor_kind=kind,
        quick_dead=qd,
    )
    diag = dict(
        revision=REVISION,
        control_rule_frames=int((control & g).sum()),
        features=names,
        anchors=an["counts"],
        fit=fitinfo,
        auc=aucs,
        quick=ds,
        quick_after_fallback=dead_stats(qd, play) if killed_model else None,
        moving_pseudo_called_dead=moving_dead,
        checks={k: bool(v) for k, v in checks},
        model_killed=killed_model,
        tau_d=ta,
        mean_p_model_by_ball_gap=gap_p,
        mean_p_model_ball_visible=(
            float(np.nanmean(p[vis_play])) if vis_play.any() else None
        ),
        seconds=time.time() - t0,
    )
    if verbose:
        print(
            f"[deadball2] revision {REVISION}; ball-control live rule on {int((control & g).sum())} graded frames"
        )
        print("[deadball2] anchors (graded frames):", an["counts"])
        print(
            f"[deadball2] out-of-fold AUC by dead-anchor kind vs live: "
            + ", ".join(
                f"{k} {('n/a' if v is None else f'{v:.3f}')}" for k, v in aucs.items()
            )
        )
        print(
            f"[deadball2] quick decode: dead fraction {ds['dead_fraction']:.4f}, stretches {ds['stretches']}, "
            f"median {ds['median_stretch_s']:.1f} s; moving-ball pseudo frames called dead {moving_dead:.4f}"
        )
        print(
            f"[deadball2] mean calibrated p_model: ball visible {diag['mean_p_model_ball_visible']}"
        )
        for k, (v, m) in gap_p.items():
            print(
                f"    ball-missing gap {k:12s} frames {m:6d}  mean p_model {('n/a' if v is None else f'{v:.3f}')}"
            )
        print(f"[deadball2] tau_d {ta['tau']:.1f} frames (window {ta['window']})")
        for k, v in checks:
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        if killed_model:
            print(
                "  -> KILL: model term dropped; dead evidence = ball rules + pre-kick-off only"
            )
            print(f"     rules-only quick decode: {diag['quick_after_fallback']}")
        else:
            print("  -> KEEP learned dead-ball model")
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
    from mtah.ballfilter import pipeline as bf_pipeline
    from mtah.evidence import compute as ev_compute
    from mtah.frames import synthetic_match

    t0 = time.time()
    cache, truth, tags = synthetic_match(seed=0)
    rng = np.random.default_rng(5)
    g = half_index() >= 0
    dead_t = g & (truth == 2)
    s, e, v = runs(dead_t.astype(np.uint8))
    for a, b in zip(
        s[v == 1], e[v == 1]
    ):  # stoppages: players walk, ball held out of play
        base = cache["players"][a].copy()
        walk = np.cumsum(rng.normal(0, 0.02, (b - a, 46, 2)), axis=0)
        cache["players"][a:b] = (base[None] + walk).astype(np.float32)
        vis = np.isfinite(cache["ball"][a:b]).all(1)
        cache["ball"][a:b][vis, 1] = -1.0
    bf = bf_pipeline(cache["ball"], half_index(), q=100.0)
    ev, _ = ev_compute(cache, bf, [a for a, _ in BLOCKS], verbose=False)
    kick = [a for a, _ in BLOCKS]
    out, diag = compute(cache, bf, ev, tags, kick)
    out2, _ = compute(cache, bf, ev, tags, kick, verbose=False)
    det = all(np.array_equal(out[k], out2[k]) for k in out)
    miss = g & ~np.isfinite(bf["pos"]).all(1)
    pd_dead = float(out["p_model"][miss & (truth == 2)].mean())
    pd_live = float(out["p_model"][miss & (truth < 2)].mean())
    qd = out["quick_dead"]
    acc = float((qd[g] == (truth[g] == 2)).mean())
    checks = [
        ("AUC out-of-bounds >= 0.85", (diag["auc"]["out_of_bounds"] or 0) >= 0.85),
        ("ball-missing dead frames: mean p_model >= 0.7", pd_dead >= 0.7),
        ("ball-missing live frames: mean p_model <= 0.3", pd_live <= 0.3),
        ("quick decode frame accuracy vs synthetic truth >= 0.9", acc >= 0.9),
        ("deterministic", det),
    ]
    print(
        f"[smoke] p_model on ball-missing frames: dead {pd_dead:.3f}, live {pd_live:.3f}; "
        f"quick-decode accuracy {acc:.4f}"
    )
    ok = True
    for k, v in checks:
        ok &= bool(v)
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"[smoke] {time.time() - t0:.1f}s -> {'PASS' if ok else 'FAIL'}")
    return ok


def run(out_path=OUT_PATH) -> int:
    from mtah import ballfilter, evidence
    from mtah.frames import (
        attack_direction,
        check_games,
        default_tags_path,
        kickoff_events,
        load_tag_restarts,
        load_tracking,
    )

    t0 = time.time()
    check_games()
    cache = load_tracking()
    for mod in (ballfilter, evidence):
        if not mod.OUT_PATH.is_file():
            raise FileNotFoundError(
                f"{mod.OUT_PATH} missing: run python -m {mod.__name__} run first"
            )
    bf, ev = ballfilter.load(), evidence.load()
    tags = load_tag_restarts(default_tags_path(), verbose=False)
    signs = attack_direction(
        cache["players"], cache["team"], half_index(), verbose=False
    )
    kick = kickoff_events(
        cache["players"], cache["team"], cache["ball"], signs, verbose=False
    ).tolist()
    print(f"[deadball2] kick-off events {kick}; restart tags {len(tags)}")
    out, diag = compute(cache, bf, ev, tags, kick)
    digest = save(out, diag, out_path)
    print(
        f"[deadball2] wrote {out_path} (+ .json)  arrays sha256 {digest[:16]}  ({time.time() - t0:.1f}s)"
    )
    return 1 if diag["model_killed"] else 0


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
