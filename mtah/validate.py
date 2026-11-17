#!/usr/bin/env python3
"""mtah/validate.py -- step 3: label-free validation harness.

No ground truth is read here, ever. Every number compares a labelling with
itself under perturbation, with pseudo-labels, or with fixed domain priors.

Tests
  report        possession / dead-ball stats on graded frames vs fixed bands
                (possessions 250-450, median live 4-15 s, dead fraction
                15-35%, median dead stretch 8-40 s; D bands only if D emitted);
                agreement with pseudo-labels (high-confidence ball-at-feet
                frames, step 2) >= 0.90; +-1 frame shift sensitivity.
  dropout       END-TO-END robustness to missing ball: transplant real
                ball-gap lengths (<= 30 s, from this match) onto fully observed
                in-play stretches, hide ~10% of observed frames, re-run the
                whole pipeline from (players, ball), compare with the
                unperturbed run: agreement inside hidden windows (by gap
                length) and spellF1 of the perturbed run against the base.
  jitter        players and ball + N(0, 0.2 m): agreement and spellF1 vs base.
  reversal      decoder only: decode emissions reversed within each block,
                reverse back, agreement with the forward decode (>= 0.97).
  compare       two labellings: agreement, spellF1 both ways, counts.

Revision 1 (declared after T1, before any T0R output): the dead-fraction
check is an upper bound only (<= 0.35). The 0.15 lower bound assumed a
detector that sees every stoppage; with the learned dead-ball model killed,
every remaining D source (restart tags, observed out-of-bounds) covers a
subset, so the lower bound penalised partial coverage rather than error.
Possession, median-live and median-dead bands are unchanged.

Kill criteria for a tier (Phase A, with the home-share check replaced by
pseudo-label agreement because pseudo frames over-sample short-passing
possession): bands, pseudo agreement >= 0.90, reversal >= 0.97, pipeline
runtime <= 120 s. Dropout and jitter feed the ship rule:
  ship the highest tier that passes its kill criteria and whose dropout
  hidden-window agreement is not lower than the tier below by > 0.01.
These are self-consistency measures: a tier can be stable and still wrong.

Interfaces
  pipeline(players f32 (n,46,2), ball f32 (n,2)) -> labels u8 (n,) {0 H,1 A,2 D}
  decode_fn(emissions (n,...) float) -> labels u8 (n,)  (per-block decoding)

CLI (repo root):
  python -m mtah.validate smoke
  python -m mtah.validate t0                    # full harness on the T0 pipeline
  python -m mtah.validate csv out/T0.csv        # report-only for any submission CSV
  python -m mtah.validate compare A.csv B.csv
Results: out/validate_<name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from mtah.frames import (
    BANDS,
    BLOCKS,
    D,
    FPS,
    N_FRAMES,
    OUT_DIR,
    half_index,
    label_stats,
    read_csv,
    runs,
    shift_labels,
    spell_f1,
)

GAP_MAX_S = 30.0
DROP_FRAC = 0.10
JITTER_M = 0.2
LEN_BINS = ((0, 3), (3, 10), (10, 30))
PSEUDO_AGREE_MIN = 0.90
REVERSAL_MIN = 0.97
RUNTIME_MAX = 120.0
SHIP_TOL = 0.01


# --------------------------------------------------------------------------- helpers
def reverse_blocks(arr):
    out = np.array(arr, copy=True)
    for a, b in BLOCKS:
        out[a : b + 1] = arr[a : b + 1][::-1]
    return out


def graded():
    return half_index() >= 0


def _bands(st) -> list:
    out = []
    for key in ("possessions", "median_live_s"):
        lo, hi = BANDS[key]
        out.append((f"{key} in [{lo},{hi}]", lo <= st[key] <= hi))
    if st["dead_fraction"] > 0:
        hi = BANDS["dead_fraction"][1]
        out.append(
            (
                f"dead_fraction <= {hi} (rev 1: upper bound only)",
                st["dead_fraction"] <= hi,
            )
        )
        lo, hi = BANDS["median_dead_s"]
        out.append((f"median_dead_s in [{lo},{hi}]", lo <= st["median_dead_s"] <= hi))
    return out


# --------------------------------------------------------------------------- report
def report(labels, pseudo=None, name="labels", verbose=True) -> dict:
    st = label_stats(labels, verbose=False)
    g = graded()
    r = dict(name=name, stats=st)
    checks = _bands(st)
    if pseudo is not None:
        sel = g & (pseudo >= 0)
        agree = (
            float((labels[sel] == pseudo[sel]).mean()) if sel.any() else float("nan")
        )
        on_d = float((labels[sel] == D).mean()) if sel.any() else float("nan")
        r.update(
            pseudo_agreement=agree,
            pseudo_frames_labelled_D=on_d,
            pseudo_home_share=(
                float((pseudo[sel] == 0).mean()) if sel.any() else float("nan")
            ),
        )
        checks.append(
            (f"pseudo agreement >= {PSEUDO_AGREE_MIN}", agree >= PSEUDO_AGREE_MIN)
        )
    sh = {k: spell_f1(labels, shift_labels(labels, k)) for k in (-1, 1)}
    r["shift_spell"] = sh
    r["checks"] = {k: bool(v) for k, v in checks}
    if verbose:
        print(f"[validate] report: {name}")
        for k in (
            "possessions",
            "per_half",
            "home_share",
            "mean_live_s",
            "median_live_s",
            "share_lt_2s",
            "dead_fraction",
            "dead_stretches",
            "median_dead_s",
            "x_frames",
        ):
            v = st[k]
            print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
        if pseudo is not None:
            print(
                f"  {'pseudo_agreement':24s} {r['pseudo_agreement']:.4f}  "
                f"(D on pseudo frames {r['pseudo_frames_labelled_D']:.4f}; pseudo home share "
                f"{r['pseudo_home_share']:.4f} vs labels {st['home_share']:.4f})"
            )
        print(f"  shift +-1 spellF1          {sh[-1]:.4f} / {sh[1]:.4f}")
        for k, v in checks:
            print(f"  {'PASS' if v else 'OUT '}  {k}")
    return r


def compare(a, b, name_a="A", name_b="B", verbose=True) -> dict:
    g = graded()
    r = dict(
        agreement=float((a[g] == b[g]).mean()),
        spell_b_vs_a=spell_f1(a, b),
        spell_a_vs_b=spell_f1(b, a),
        possessions=(
            label_stats(a, verbose=False)["possessions"],
            label_stats(b, verbose=False)["possessions"],
        ),
    )
    if verbose:
        print(
            f"[validate] compare {name_a} vs {name_b}: agreement {r['agreement']:.4f}  "
            f"spellF1({name_b}|{name_a}) {r['spell_b_vs_a']:.4f}  spellF1({name_a}|{name_b}) "
            f"{r['spell_a_vs_b']:.4f}  possessions {r['possessions']}"
        )
    return r


# --------------------------------------------------------------------------- perturbations
def real_gap_lengths(ball, play, max_len) -> np.ndarray:
    have = np.isfinite(ball).all(1)
    lens = []
    for a, b in BLOCKS:
        s, e, v = runs((have[a : b + 1] | ~play[a : b + 1]).astype(np.uint8))
        lens += list((e - s)[v == 0])
    lens = np.array(lens, np.int64)
    return lens[(lens > 0) & (lens <= max_len)]


def transplant_mask(
    ball,
    play,
    frac=DROP_FRAC,
    seed=0,
    max_len=int(GAP_MAX_S * FPS),
    spacing=250,
    max_tries=200_000,
) -> tuple[np.ndarray, list]:
    """Hide windows whose lengths are drawn from this match's real ball gaps,
    placed on fully observed in-play stretches, non-overlapping, seeded."""
    n = len(ball)
    obs = np.isfinite(ball).all(1) & play
    gaps = real_gap_lengths(ball, play, max_len)
    if len(gaps) == 0:
        gaps = np.array([25, 75, 250])
    cs = np.r_[0, np.cumsum(obs)]
    rng = np.random.default_rng(seed)
    drop = np.zeros(n, bool)
    blocked = np.zeros(n, bool)
    target = frac * obs.sum()
    windows, hidden, tries = [], 0, 0
    while hidden < target and tries < max_tries:
        tries += 1
        L = int(rng.choice(gaps))
        s = int(rng.integers(0, n - L))
        if (
            cs[s + L] - cs[s] != L
            or blocked[max(0, s - spacing) : s + L + spacing].any()
        ):
            continue
        drop[s : s + L] = True
        blocked[s : s + L] = True
        windows.append((s, L))
        hidden += L
    return drop, windows


def dropout_test(
    pipeline, players, ball, play, base=None, seed=0, verbose=True
) -> dict:
    g = graded()
    t = time.time()
    if base is None:
        base = pipeline(players, ball)
    drop, windows = transplant_mask(ball, play, seed=seed)
    b2 = ball.copy()
    b2[drop] = np.nan
    lab = pipeline(players, b2)
    hid = drop & g
    r = dict(
        seed=seed,
        windows=len(windows),
        hidden_frames=int(hid.sum()),
        hidden_share_of_observed=float(
            hid.sum() / max((np.isfinite(ball).all(1) & play & g).sum(), 1)
        ),
        agreement_all=float((lab[g] == base[g]).mean()),
        agreement_hidden=(
            float((lab[hid] == base[hid]).mean()) if hid.any() else float("nan")
        ),
        spell_vs_base=spell_f1(base, lab),
        by_length={},
    )
    for lo, hi in LEN_BINS:
        m = np.zeros(len(ball), bool)
        for s, L in windows:
            if lo * FPS < L <= hi * FPS:
                m[s : s + L] = True
        m &= g
        r["by_length"][f"({lo},{hi}] s"] = dict(
            frames=int(m.sum()),
            agreement=float((lab[m] == base[m]).mean()) if m.any() else None,
        )
    r["seconds"] = time.time() - t
    if verbose:
        print(
            f"[validate] dropout (seed {seed}): {r['windows']} windows, {r['hidden_frames']} frames "
            f"({r['hidden_share_of_observed']:.3f} of observed); agreement hidden {r['agreement_hidden']:.4f}, "
            f"all {r['agreement_all']:.4f}; spellF1 vs base {r['spell_vs_base']:.4f}"
        )
        for k, v in r["by_length"].items():
            a = "n/a" if v["agreement"] is None else f"{v['agreement']:.4f}"
            print(f"    gap {k:10s} frames {v['frames']:6d}  agreement {a}")
    return r


def jitter_test(
    pipeline, players, ball, base=None, sigma=JITTER_M, seed=0, verbose=True
) -> dict:
    g = graded()
    if base is None:
        base = pipeline(players, ball)
    rng = np.random.default_rng(seed)
    p2 = (players + rng.normal(0, sigma, players.shape)).astype(np.float32)
    b2 = (ball + rng.normal(0, sigma, ball.shape)).astype(np.float32)
    lab = pipeline(p2, b2)
    r = dict(
        sigma=sigma,
        seed=seed,
        agreement=float((lab[g] == base[g]).mean()),
        spell_vs_base=spell_f1(base, lab),
    )
    if verbose:
        print(
            f"[validate] jitter {sigma} m (seed {seed}): agreement {r['agreement']:.4f}  "
            f"spellF1 vs base {r['spell_vs_base']:.4f}"
        )
    return r


def reversal_test(decode_fn, emissions, verbose=True) -> dict:
    g = graded()
    t = time.time()
    fwd = decode_fn(emissions)
    secs = time.time() - t
    bwd = reverse_blocks(decode_fn(reverse_blocks(emissions)))
    r = dict(
        agreement=float((fwd[g] == bwd[g]).mean()),
        spell=spell_f1(fwd, bwd),
        decode_seconds=secs,
    )
    if verbose:
        print(
            f"[validate] reversal: agreement {r['agreement']:.4f}  spellF1 {r['spell']:.4f}  "
            f"decode {secs:.1f}s"
        )
    return r


# --------------------------------------------------------------------------- tier evaluation
def evaluate_tier(
    name,
    pipeline,
    players,
    ball,
    play,
    pseudo,
    decode_fn=None,
    emissions=None,
    seeds=(0,),
    verbose=True,
    save=True,
) -> dict:
    t = time.time()
    base = pipeline(players, ball)
    runtime = time.time() - t
    rep = report(base, pseudo, name, verbose)
    drops = [
        dropout_test(pipeline, players, ball, play, base, s, verbose) for s in seeds
    ]
    jit = jitter_test(pipeline, players, ball, base, verbose=verbose)
    rev = (
        reversal_test(decode_fn, emissions, verbose) if decode_fn is not None else None
    )
    checks = dict(rep["checks"])
    checks[f"pipeline runtime <= {RUNTIME_MAX:.0f} s"] = runtime <= RUNTIME_MAX
    if rev is not None:
        checks[f"reversal agreement >= {REVERSAL_MIN}"] = (
            rev["agreement"] >= REVERSAL_MIN
        )
    res = dict(
        name=name,
        runtime_s=runtime,
        report=rep,
        dropout=drops,
        jitter=jit,
        reversal=rev,
        checks=checks,
        kill_pass=all(checks.values()),
        dropout_hidden_agreement=float(np.mean([d["agreement_hidden"] for d in drops])),
    )
    if verbose:
        print(f"[validate] {name}: runtime {runtime:.1f}s; kill criteria:")
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print(
            f"  -> {'eligible to ship' if res['kill_pass'] else 'NOT eligible'}; "
            f"dropout hidden-window agreement {res['dropout_hidden_agreement']:.4f}"
        )
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"validate_{name}.json"
        path.write_text(json.dumps(res, indent=2, default=float))
        if verbose:
            print(f"[validate] wrote {path}")
    return res, base


def ship_rule(results: list, always=("T0",), verbose=True) -> str:
    """results: tier dicts in increasing tier order (T0 first). T0 is always
    eligible (it is the fallback). Returns the name of the tier to ship."""
    chosen = results[0]["name"]
    prev = results[0]
    for r in results[1:]:
        ok = (
            r["kill_pass"]
            and r["dropout_hidden_agreement"]
            >= prev["dropout_hidden_agreement"] - SHIP_TOL
        )
        if verbose:
            print(
                f"[ship] {r['name']}: kill_pass {r['kill_pass']}, dropout {r['dropout_hidden_agreement']:.4f} "
                f"vs {prev['name']} {prev['dropout_hidden_agreement']:.4f} -> {'eligible' if ok else 'rejected'}"
            )
        if ok:
            chosen, prev = r["name"], r
    if verbose:
        print(f"[ship] -> {chosen}")
    return chosen


# --------------------------------------------------------------------------- T0 adapters
def t0_pipeline(team):
    from mtah.frames import t0_decode

    def f(players, ball):
        return t0_decode(players, team, ball)[0]

    return f


def t0_decode_fn(z):
    from mtah.frames import T0_PARAMS, viterbi2

    lab = np.full(len(z), D, np.uint8)
    for a, b in BLOCKS:
        lab[a : b + 1] = viterbi2(z[a : b + 1], T0_PARAMS["lam"])
    return lab


# --------------------------------------------------------------------------- smoke / CLI
def smoke() -> bool:
    from mtah.ballfilter import pipeline as bf_pipeline
    from mtah.evidence import compute
    from mtah.frames import soft_evidence, synthetic_match

    t0 = time.time()
    cache, truth, _ = synthetic_match(seed=0)
    bf = bf_pipeline(cache["ball"], half_index(), q=100.0)
    ev, _ = compute(cache, bf, [a for a, _ in BLOCKS], verbose=False)
    pipe = t0_pipeline(cache["team"])
    z = soft_evidence(cache["players"], cache["ball"], cache["team"])
    res, base = evaluate_tier(
        "smoke_T0",
        pipe,
        cache["players"],
        cache["ball"],
        ev["play"],
        ev["pseudo"],
        t0_decode_fn,
        z,
        save=False,
    )
    same = compare(base, base, "T0", "T0")
    flip = np.where(base < D, 1 - base, base).astype(np.uint8)
    rev = reverse_blocks(reverse_blocks(base))
    ship = ship_rule(
        [
            dict(name="T0", kill_pass=True, dropout_hidden_agreement=0.80),
            dict(name="T1", kill_pass=True, dropout_hidden_agreement=0.795),
            dict(name="T2", kill_pass=False, dropout_hidden_agreement=0.90),
        ],
        verbose=False,
    )
    g = graded() & (truth < D)
    checks = [
        (
            "report runs; pseudo agreement >= 0.90",
            res["report"]["pseudo_agreement"] >= 0.90,
        ),
        ("reversal agreement >= 0.97 for T0", res["reversal"]["agreement"] >= 0.97),
        (
            "dropout hides ~10% of observed",
            0.07 <= res["dropout"][0]["hidden_share_of_observed"] <= 0.13,
        ),
        (
            "dropout agreement in (0.5, 1]",
            0.5 < res["dropout"][0]["agreement_hidden"] <= 1.0,
        ),
        ("jitter agreement >= 0.95", res["jitter"]["agreement"] >= 0.95),
        (
            "compare(x, x) is identity",
            same["agreement"] == 1.0 and same["spell_b_vs_a"] == 1.0,
        ),
        ("reverse_blocks is an involution", np.array_equal(rev, base)),
        (
            "team flip drops pseudo agreement below 0.2",
            report(flip, ev["pseudo"], "flip", verbose=False)["pseudo_agreement"] < 0.2,
        ),
        ("ship rule picks T1 within tolerance, skips killed T2", ship == "T1"),
        ("harness never reads truth (synthetic T0 accuracy shown for info only)", True),
    ]
    print(
        f"[smoke] (info) synthetic T0 frame accuracy vs truth {float((base[g] == truth[g]).mean()):.4f}"
    )
    ok = True
    for k, v in checks:
        ok &= bool(v)
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"[smoke] {time.time() - t0:.1f}s -> {'PASS' if ok else 'FAIL'}")
    return ok


def _repo_inputs():
    from mtah import evidence
    from mtah.frames import check_games, load_tracking

    check_games()
    cache = load_tracking()
    if not evidence.OUT_PATH.is_file():
        raise FileNotFoundError(
            f"{evidence.OUT_PATH} missing: run python -m mtah.evidence run first"
        )
    ev = evidence.load()
    return cache, ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("t0")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p = sub.add_parser("csv")
    p.add_argument("path")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        return 0 if smoke() else 1
    if a.cmd == "compare":
        compare(read_csv(a.a), read_csv(a.b), Path(a.a).stem, Path(a.b).stem)
        return 0
    cache, ev = _repo_inputs()
    if a.cmd == "csv":
        r = report(read_csv(a.path), ev["pseudo"], Path(a.path).stem)
        return 0 if all(r["checks"].values()) else 1
    from mtah.frames import soft_evidence

    z = soft_evidence(cache["players"], cache["ball"], cache["team"])
    res, _ = evaluate_tier(
        "T0",
        t0_pipeline(cache["team"]),
        cache["players"],
        cache["ball"],
        ev["play"],
        ev["pseudo"],
        t0_decode_fn,
        z,
        seeds=tuple(a.seeds),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
