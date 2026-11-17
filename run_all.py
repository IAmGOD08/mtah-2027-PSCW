#!/usr/bin/env python3
"""run_all.py -- regenerate the submission from raw data with one command.

    python run_all.py [--from-raw] [--smoke] [--expect SHA256]

Each step is a module CLI run in a subprocess (full output in out/run_all.log).
Accepted exit codes encode pre-registered kill decisions, not errors:
  frames selftest          0        reference metric on analytic cases
  frames ship-t0           0        T0 + parity with level0 and baseline writer
  ballfilter run           0 | 1    1 = Kalman killed -> linear fallback saved
  evidence run             0 | 2    2 = pseudo-label yield rule killed T2
  validate t0              0        label-free harness on T0
  deadball2 run            0 | 1    1 = learned model killed -> rules-only D
  hsmm t1                  0        T1 (exit 1 = parity failure: fatal)
  restarts tier            0        T0R, ship rule, out/submission.csv
then evaluation/validate_submission.py on out/submission.csv (no ground truth).

--from-raw  delete cache/tracking.npz first, so the ATD file is re-parsed.
--smoke     also run every module's synthetic smoke test first.
--expect    fail unless out/submission.csv has this sha256.
Determinism: OMP_NUM_THREADS=1 (scikit-learn), PYTHONHASHSEED=0, fixed seeds in
every module. The public ground truth is never read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
CACHE = ROOT / "cache"

SMOKES = [
    ("smoke frames", "mtah.frames", ["smoke"]),
    ("smoke ballfilter", "mtah.ballfilter", ["smoke"]),
    ("smoke evidence", "mtah.evidence", ["smoke"]),
    ("smoke validate", "mtah.validate", ["smoke"]),
    ("smoke deadball2", "mtah.deadball2", ["smoke"]),
    ("smoke hsmm", "mtah.hsmm", ["smoke"]),
    ("smoke restarts", "mtah.restarts", ["smoke"]),
]
STEPS = [
    ("frames selftest", "mtah.frames", ["selftest"], {0}),
    ("T0", "mtah.frames", ["ship-t0"], {0}),
    ("ballfilter", "mtah.ballfilter", ["run"], {0, 1}),
    ("evidence", "mtah.evidence", ["run"], {0, 2}),
    ("validate T0", "mtah.validate", ["t0"], {0}),
    ("deadball2", "mtah.deadball2", ["run"], {0, 1}),
    ("T1", "mtah.hsmm", ["t1"], {0}),
    ("T0R + ship rule", "mtah.restarts", ["tier"], {0}),
]
STALE = ["SHIP.json", "submission.csv"]


def env():
    e = dict(os.environ)
    e.setdefault("OMP_NUM_THREADS", "1")
    e["PYTHONHASHSEED"] = "0"
    return e


def run(name, argv, ok, log, tail=10):
    t = time.time()
    p = subprocess.run(
        [sys.executable, *argv], cwd=ROOT, env=env(), capture_output=True, text=True
    )
    secs = time.time() - t
    text = p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
    log.write(
        f"\n===== {name}: {' '.join(argv)} (exit {p.returncode}, {secs:.1f}s) =====\n{text}"
    )
    log.flush()
    lines = [l for l in p.stdout.rstrip().splitlines() if l.strip()]
    print(f"\n[{name}] exit {p.returncode} in {secs:.1f}s")
    for l in lines[-tail:]:
        print("   " + l)
    if p.returncode not in ok:
        print(p.stderr[-2000:])
        raise SystemExit(
            f"[run_all] step '{name}' failed with exit {p.returncode} (allowed {sorted(ok)}); "
            f"see {OUT / 'run_all.log'}"
        )
    return p.returncode, secs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def versions():
    out = dict(python=platform.python_version(), platform=platform.platform())
    for m in ("numpy", "scipy", "sklearn", "pandas", "numba"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    try:
        out["git_head"] = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip()
            or None
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "mtah", "run_all.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        out["git_dirty_code"] = bool(dirty)
    except Exception:
        out["git_head"], out["git_dirty_code"] = None, None
    return out


def report(codes, timings, sub_sha, official):
    bf = jload(CACHE / "ballfilter.json")
    ev = jload(CACHE / "evidence.json")
    db = jload(CACHE / "deadball2.json")
    ship = jload(OUT / "SHIP.json")
    tiers = {n: jload(OUT / f"validate_{n}.json") for n in ("T0", "T1", "T0R")}
    print(
        "\n"
        + "=" * 78
        + "\nDIAGNOSTIC REPORT (label-free; no ground truth read)\n"
        + "=" * 78
    )
    d = bf.get("diag", {})
    print(
        f"step 1 ball track : {bf.get('mode')}  r {d.get('r_m', float('nan')):.3f} m  q {d.get('q')}  "
        f"rejected {d.get('rejected_share', float('nan')):.4f}  failed checks "
        f"{[k for k, v in d.get('checks', {}).items() if not v]}"
    )
    print(
        f"step 2 evidence   : play starts {ev.get('play_starts')}  tau {ev.get('tau', {}).get('tau', float('nan')):.2f}"
        f"  pseudo yield {ev.get('pseudo_yield', float('nan')):.4f}  T2 alive {ev.get('t2_alive')}"
    )
    print(
        f"step 4 dead ball  : model killed {db.get('model_killed')} (revision {db.get('revision')})  "
        f"AUC oob {db.get('auc', {}).get('out_of_bounds')}  failed checks "
        f"{[k for k, v in db.get('checks', {}).items() if not v]}"
    )
    print(
        "\ntier   poss  median_live  dead_frac  median_dead  pseudo_agr  dropout  reversal  kill_pass"
    )
    for n, t in tiers.items():
        if not t:
            print(f"{n:5s}  (missing)")
            continue
        st = t["report"]["stats"]
        rv = (t.get("reversal") or {}).get("agreement", float("nan"))
        print(
            f"{n:5s} {st['possessions']:5d}  {st['median_live_s']:11.2f}  {st['dead_fraction']:9.4f}  "
            f"{st['median_dead_s']:11.2f}  {t['report'].get('pseudo_agreement', float('nan')):10.4f}  "
            f"{t['dropout_hidden_agreement']:7.4f}  {rv:8.4f}  {t['kill_pass']}"
        )
        failed = [k for k, v in t.get("checks", {}).items() if not v]
        if failed:
            print(f"       failed: {failed}")
    print(f"\nship rule -> {ship.get('chosen')}   out/submission.csv sha256 {sub_sha}")
    print(f"official validate_submission.py: {official}")
    print("timings: " + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from-raw", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--expect", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    for f in STALE:
        (OUT / f).unlink(missing_ok=True)
    if a.from_raw:
        (CACHE / "tracking.npz").unlink(missing_ok=True)
        print("[run_all] removed cache/tracking.npz: the ATD file will be re-parsed")
    codes, timings = {}, {}
    with open(OUT / "run_all.log", "w") as log:
        if a.smoke:
            for name, mod, args in SMOKES:
                codes[name], timings[name] = run(
                    name, ["-m", mod, *args], {0}, log, tail=2
                )
        for name, mod, args, ok in STEPS:
            codes[name], timings[name] = run(name, ["-m", mod, *args], ok, log)
        sub = OUT / "submission.csv"
        if not sub.is_file():
            raise SystemExit("[run_all] out/submission.csv was not written")
        sub_sha = sha256(sub)
        validator = ROOT / "evaluation" / "validate_submission.py"
        if validator.is_file():
            rc, _ = run(
                "validate_submission", [str(validator), str(sub)], {0, 1}, log, tail=6
            )
            official = "PASS" if rc == 0 else f"exit {rc} (see log)"
        else:
            official = "not found"
    manifest = dict(
        versions=versions(),
        exit_codes=codes,
        timings_s=timings,
        ship=jload(OUT / "SHIP.json").get("chosen"),
        submission_sha256=sub_sha,
        expected_sha256=a.expect,
        official_validator=official,
        seconds=time.time() - t0,
    )
    (OUT / "run_all.json").write_text(json.dumps(manifest, indent=2))
    report(codes, timings, sub_sha, official)
    print(
        f"\n[run_all] manifest out/run_all.json; full log out/run_all.log; total {time.time() - t0:.0f}s"
    )
    if a.expect and a.expect != sub_sha:
        print(f"[run_all] FAIL: sha256 {sub_sha} != expected {a.expect}")
        return 1
    if a.expect:
        print("[run_all] PASS: submission sha256 matches --expect")
    return 0 if official in ("PASS", "not found") else 1


if __name__ == "__main__":
    sys.exit(main())
